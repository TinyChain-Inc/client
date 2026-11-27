use std::pin::Pin;

use base64::Engine as _;
use bytes::Bytes;
use destream::en::{self, Error as _, IntoStream};
use futures::{Stream, StreamExt, TryStreamExt, stream};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use super::{Budget, Error};

type ObjectFuture<'en> = futures::future::Either<
    futures::future::Ready<Result<Py<PyAny>, Error>>,
    Pin<Box<dyn std::future::Future<Output = Result<Py<PyAny>, Error>> + Send + 'en>>,
>;

#[derive(Clone)]
struct Encoder(Budget);

fn ready<'en>(value: Py<PyAny>) -> ObjectFuture<'en> {
    futures::future::Either::Left(futures::future::ready(Ok(value)))
}

pub(super) async fn encode<'en>(
    value: impl IntoStream<'en>,
    budget: Budget,
) -> Result<Py<PyAny>, Error> {
    value.into_stream(Encoder(budget))?.await
}

macro_rules! primitives {
    ($($method:ident($ty:ty)),* $(,)?) => {$ (
        fn $method(self, value: $ty) -> Result<Self::Ok, Error> {
            self.0.charge(std::mem::size_of::<$ty>() + std::mem::size_of::<Py<PyAny>>())?;
            Python::attach(|py| Ok(ready(value.into_py_any(py)?)))
        }
    )*};
}

macro_rules! arrays {
    ($($method:ident($ty:ty)),* $(,)?) => {$ (
        fn $method<T, S>(self, chunks: S) -> Result<Self::Ok, Error>
        where T: IntoIterator<Item = $ty> + Send + Unpin + 'en,
            S: Stream<Item = T> + Send + Unpin + 'en,
            T::IntoIter: Send + Unpin + 'en {
            self.encode_seq_stream(chunks.flat_map(stream::iter))
        }
    )*};
}

impl<'en> en::Encoder<'en> for Encoder {
    type Ok = ObjectFuture<'en>;
    type Error = Error;
    type EncodeMap = Map<'en>;
    type EncodeSeq = Sequence<'en>;
    type EncodeTuple = Sequence<'en>;

    primitives!(
        encode_bool(bool),
        encode_i8(i8),
        encode_i16(i16),
        encode_i32(i32),
        encode_i64(i64),
        encode_u8(u8),
        encode_u16(u16),
        encode_u32(u32),
        encode_u64(u64)
    );

    fn encode_f32(self, value: f32) -> Result<Self::Ok, Error> {
        self.encode_f64(f64::from(value))
    }

    fn encode_f64(self, value: f64) -> Result<Self::Ok, Error> {
        if !value.is_finite() {
            return Err(Error::custom(
                "JSON encoding does not support nonfinite floating-point values",
            ));
        }

        self.0
            .charge(std::mem::size_of::<f64>() + std::mem::size_of::<Py<PyAny>>())?;

        Python::attach(|py| Ok(ready(value.into_py_any(py)?)))
    }

    fn encode_str<S: AsRef<str> + Send + 'en>(self, value: S) -> Result<Self::Ok, Error> {
        let text = value.as_ref();
        self.0
            .charge(text.len() + std::mem::size_of::<Py<PyAny>>())?;

        Python::attach(|py| Ok(ready(PyString::new(py, text).into_any().unbind())))
    }

    fn encode_none(self) -> Result<Self::Ok, Error> {
        self.0.charge(std::mem::size_of::<Py<PyAny>>())?;
        Python::attach(|py| Ok(ready(py.None())))
    }

    fn encode_unit(self) -> Result<Self::Ok, Error> {
        self.encode_none()
    }

    fn encode_some<T: IntoStream<'en> + 'en>(self, value: T) -> Result<Self::Ok, Error> {
        value.into_stream(self)
    }

    fn encode_bytes<B: Into<Bytes>>(self, value: B) -> Result<Self::Ok, Error> {
        let value = value.into();
        let len = base64::encoded_len(value.len(), true)
            .ok_or_else(|| Error::custom("base64 size overflow"))?;

        self.0.charge(len)?;
        self.encode_str(base64::engine::general_purpose::STANDARD.encode(value))
    }

    fn collect_bytes<B: IntoIterator<Item = u8>>(self, bytes: B) -> Result<Self::Ok, Error> {
        let mut value = Vec::new();
        for byte in bytes {
            self.0.charge(1)?;
            value.push(byte);
        }

        self.encode_bytes(value)
    }

    fn encode_uuid(self, value: uuid::Uuid) -> Result<Self::Ok, Error> {
        self.collect_str(&value)
    }

    fn encode_seq(self, _: Option<usize>) -> Result<Sequence<'en>, Error> {
        Ok(Sequence {
            encoder: self,
            children: Vec::new(),
        })
    }

    fn encode_tuple(self, len: usize) -> Result<Sequence<'en>, Error> {
        self.encode_seq(Some(len))
    }

    fn encode_map(self, _: Option<usize>) -> Result<Map<'en>, Error> {
        Ok(Map {
            encoder: self,
            children: Vec::new(),
            pending: None,
        })
    }

    fn encode_seq_stream<T, S>(self, source: S) -> Result<Self::Ok, Error>
    where
        T: IntoStream<'en> + 'en,
        S: Stream<Item = T> + Send + Unpin + 'en,
    {
        let encoder = self.clone();
        let mut source = source.map(move |value| value.into_stream(encoder.clone()));
        Ok(futures::future::Either::Right(Box::pin(async move {
            let list = Python::attach(|py| PyList::empty(py).unbind());
            while let Some(value) = source.try_next().await? {
                let value = value.await?;
                self.0.charge(std::mem::size_of::<Py<PyAny>>())?;
                Python::attach(|py| list.bind(py).append(value)).map_err(Error::from)?;
            }

            Ok(list.into_any())
        })))
    }

    fn encode_map_stream<K, V, S>(self, source: S) -> Result<Self::Ok, Error>
    where
        K: IntoStream<'en> + 'en,
        V: IntoStream<'en> + 'en,
        S: Stream<Item = (K, V)> + Send + Unpin + 'en,
    {
        let encoder = self.clone();
        let mut source = source.map(move |(key, value)| {
            Ok::<_, Error>((
                key.into_stream(encoder.clone())?,
                value.into_stream(encoder.clone())?,
            ))
        });

        Ok(futures::future::Either::Right(Box::pin(async move {
            let map = Python::attach(|py| PyDict::new(py).unbind());
            while let Some((key, value)) = source.try_next().await? {
                let key = key.await?;
                let value = value.await?;
                self.0.charge(2 * std::mem::size_of::<Py<PyAny>>())?;
                Python::attach(|py| map.bind(py).set_item(key, value)).map_err(Error::from)?;
            }

            Ok(map.into_any())
        })))
    }

    fn encode_events<T, S>(self, events: S) -> Result<Self::Ok, Error>
    where
        T: IntoStream<'en> + 'en,
        S: Stream<Item = Result<en::Event<T>, Error>> + Send + 'en,
    {
        let encoder = self.clone();
        let events = events.map(move |event| {
            Ok::<_, Error>(match event? {
                en::Event::Value(value) => en::Event::Value(value.into_stream(encoder.clone())?),
                en::Event::SeqStart(len) => en::Event::SeqStart(len),
                en::Event::MapStart(len) => en::Event::MapStart(len),
                en::Event::End => en::Event::End,
            })
        });

        Ok(futures::future::Either::Right(Box::pin(async move {
            let mut events = std::pin::pin!(events);
            let mut frames = Vec::new();
            let mut root = None;
            while let Some(event) = events.try_next().await? {
                let value = match event {
                    en::Event::SeqStart(_) => {
                        self.0.charge(std::mem::size_of::<Frame>())?;
                        frames.push(Frame::Sequence(Python::attach(|py| {
                            PyList::empty(py).unbind()
                        })));

                        continue;
                    }
                    en::Event::MapStart(_) => {
                        self.0.charge(std::mem::size_of::<Frame>())?;
                        frames.push(Frame::Map(
                            Python::attach(|py| PyDict::new(py).unbind()),
                            None,
                        ));

                        continue;
                    }
                    en::Event::Value(value) => value.await?,
                    en::Event::End => match frames
                        .pop()
                        .ok_or_else(|| Error::custom("unexpected container end"))?
                    {
                        Frame::Sequence(value) => value.into_any(),
                        Frame::Map(value, None) => value.into_any(),
                        Frame::Map(_, Some(_)) => {
                            return Err(Error::custom("missing map value"));
                        }
                    },
                };

                if let Some(frame) = frames.last_mut() {
                    self.0.charge(std::mem::size_of::<Py<PyAny>>())?;
                    frame.push(value)?;
                } else if root.replace(value).is_some() {
                    return Err(Error::custom("multiple root values"));
                }
            }

            if !frames.is_empty() {
                return Err(Error::custom("unterminated container"));
            }

            root.ok_or_else(|| Error::custom("missing root value"))
        })))
    }

    arrays!(
        encode_array_bool(bool),
        encode_array_i8(i8),
        encode_array_i16(i16),
        encode_array_i32(i32),
        encode_array_i64(i64),
        encode_array_u8(u8),
        encode_array_u16(u16),
        encode_array_u32(u32),
        encode_array_u64(u64),
        encode_array_f32(f32),
        encode_array_f64(f64)
    );
}

enum Frame {
    Sequence(Py<PyList>),
    Map(Py<PyDict>, Option<Py<PyAny>>),
}

impl Frame {
    fn push(&mut self, value: Py<PyAny>) -> Result<(), Error> {
        Python::attach(|py| {
            match self {
                Self::Sequence(list) => list.bind(py).append(value)?,
                Self::Map(map, key) => {
                    if let Some(key) = key.take() {
                        map.bind(py).set_item(key, value)?;
                    } else {
                        *key = Some(value);
                    }
                }
            }

            Ok(())
        })
    }
}

struct Sequence<'en> {
    encoder: Encoder,
    children: Vec<ObjectFuture<'en>>,
}

impl<'en> en::EncodeSeq<'en> for Sequence<'en> {
    type Ok = ObjectFuture<'en>;
    type Error = Error;

    fn encode_element<T: IntoStream<'en> + 'en>(&mut self, value: T) -> Result<(), Error> {
        self.encoder
            .0
            .charge(std::mem::size_of::<ObjectFuture<'en>>())?;

        self.children.push(value.into_stream(self.encoder.clone())?);
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Error> {
        Ok(futures::future::Either::Right(Box::pin(async move {
            let list = Python::attach(|py| PyList::empty(py).unbind());
            for child in self.children {
                let value = child.await?;
                Python::attach(|py| list.bind(py).append(value)).map_err(Error::from)?;
            }

            Ok(list.into_any())
        })))
    }
}

impl<'en> en::EncodeTuple<'en> for Sequence<'en> {
    type Ok = ObjectFuture<'en>;
    type Error = Error;

    fn encode_element<T: IntoStream<'en> + 'en>(&mut self, value: T) -> Result<(), Error> {
        en::EncodeSeq::encode_element(self, value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        en::EncodeSeq::end(self)
    }
}

struct Map<'en> {
    encoder: Encoder,
    children: Vec<(ObjectFuture<'en>, ObjectFuture<'en>)>,
    pending: Option<ObjectFuture<'en>>,
}

impl<'en> en::EncodeMap<'en> for Map<'en> {
    type Ok = ObjectFuture<'en>;
    type Error = Error;

    fn encode_key<T: IntoStream<'en> + 'en>(&mut self, value: T) -> Result<(), Error> {
        if self.pending.is_some() {
            return Err(Error::custom("missing map value"));
        }

        self.encoder
            .0
            .charge(std::mem::size_of::<ObjectFuture<'en>>())?;

        self.pending = Some(value.into_stream(self.encoder.clone())?);
        Ok(())
    }

    fn encode_value<T: IntoStream<'en> + 'en>(&mut self, value: T) -> Result<(), Error> {
        if self.pending.is_none() {
            return Err(Error::custom("missing map key"));
        }

        self.encoder
            .0
            .charge(std::mem::size_of::<ObjectFuture<'en>>())?;

        let value = value.into_stream(self.encoder.clone())?;
        let key = self
            .pending
            .take()
            .expect("map key checked before encoding value");

        self.children.push((key, value));
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Error> {
        if self.pending.is_some() {
            return Err(Error::custom("missing map value"));
        }

        Ok(futures::future::Either::Right(Box::pin(async move {
            let map = Python::attach(|py| PyDict::new(py).unbind());
            for (key, value) in self.children {
                let key = key.await?;
                let value = value.await?;
                Python::attach(|py| map.bind(py).set_item(key, value)).map_err(Error::from)?;
            }

            Ok(map.into_any())
        })))
    }
}
