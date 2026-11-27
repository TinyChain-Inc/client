use std::collections::VecDeque;
use std::marker::PhantomData;

use destream::de::{
    self, Container, Decoder as _, Error as _, FromStream, Inspection, Kind, Slot, Visitor,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use super::{Budget, Error};

pub(crate) struct Decoder {
    current: Option<Py<PyAny>>,
    frames: Vec<VecDeque<Py<PyAny>>>,
    budget: Budget,
}

impl Decoder {
    pub(crate) fn new(value: Py<PyAny>, budget: Budget) -> Self {
        Self {
            current: Some(value),
            frames: Vec::new(),
            budget,
        }
    }

    pub(crate) fn finish(&self) -> Result<(), Error> {
        if self.current.is_none() && self.frames.is_empty() {
            Ok(())
        } else {
            Err(Error::custom("unconsumed Python wire data"))
        }
    }

    fn take(&mut self) -> Result<Py<PyAny>, Error> {
        self.budget.charge(std::mem::size_of::<Py<PyAny>>())?;
        self.current
            .take()
            .ok_or_else(|| Error::custom("expected Python wire value"))
    }
}

macro_rules! scalar_decoders {
    ($($method:ident),* $(,)?) => {$ (
        async fn $method<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
            self.decode_any(visitor).await
        }
    )*};
}

macro_rules! array_decoders {
    ($($method:ident => $visit:ident($ty:ty)),* $(,)?) => {$ (
        async fn $method<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
            let container = self.open_container(Kind::Seq, None).await?;
            visitor.$visit(Access::<$ty> { decoder: self, container, marker: PhantomData }).await
        }
    )*};
}

macro_rules! numeric_decoders {
    ($($method:ident => $visit:ident($ty:ty)),* $(,)?) => {$ (
        async fn $method<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
            let raw = self.take()?;
            Python::attach(|py| visitor.$visit(raw.bind(py).extract::<$ty>()?))
        }
    )*};
}

impl de::Decoder for Decoder {
    type Error = Error;

    async fn peek_kind(&mut self) -> Result<Kind, Error> {
        Python::attach(|py| {
            let value = self
                .current
                .as_ref()
                .ok_or_else(|| Error::custom("missing wire value"))?
                .bind(py);

            Ok(if value.is_instance_of::<PyDict>() {
                Kind::Map
            } else if value.is_instance_of::<PyList>()
                || value.is_instance_of::<PyTuple>()
                || value.is_instance_of::<PyComplex>()
            {
                Kind::Seq
            } else {
                Kind::Leaf
            })
        })
    }

    async fn open_container(
        &mut self,
        kind: Kind,
        hint: Option<usize>,
    ) -> Result<Container, Error> {
        if self.frames.len() >= tc_value::MAX_VALUE_DEPTH {
            return Err(Error::custom("Python wire nesting exceeds limit"));
        }

        let raw = self.take()?;
        let mut children = Python::attach(|py| -> Result<VecDeque<Py<PyAny>>, Error> {
            let raw = raw.bind(py);
            let count = if raw.is_instance_of::<PyComplex>() {
                2
            } else {
                raw.len()?
            };

            let slots = count
                .checked_mul(if kind == Kind::Map { 2 } else { 1 })
                .ok_or_else(|| Error::custom("container size overflow"))?;

            self.budget.charge(
                slots
                    .checked_mul(std::mem::size_of::<Py<PyAny>>())
                    .ok_or_else(|| Error::custom("container size overflow"))?,
            )?;

            let mut children = VecDeque::with_capacity(slots);
            if kind == Kind::Map {
                for (key, value) in raw.cast::<PyDict>()?.iter() {
                    if !key.is_instance_of::<PyString>() {
                        return Err(Error::custom("wire map keys must be strings"));
                    }

                    children.push_back(key.unbind());
                    children.push_back(value.unbind());
                }
            } else if kind == Kind::Seq {
                if let Ok(list) = raw.cast::<PyList>() {
                    children.extend(list.iter().map(Bound::unbind));
                } else if let Ok(n) = raw.cast::<PyComplex>() {
                    children.push_back(n.real().into_pyobject(py)?.into_any().unbind());
                    children.push_back(n.imag().into_pyobject(py)?.into_any().unbind());
                } else {
                    children.extend(raw.cast::<PyTuple>()?.iter().map(Bound::unbind));
                }
            } else {
                return Err(Error::custom("expected container"));
            }

            Ok(children)
        })?;

        let empty = children.is_empty();
        self.current = children.pop_front();
        if !empty {
            self.frames.push(children);
        }

        Container::new(kind, hint, empty)
    }

    async fn finish_child(&mut self, container: &mut Container) -> Result<(), Error> {
        if self.current.is_some() {
            return Err(Error::custom("child was not consumed"));
        }

        let frame = self
            .frames
            .last_mut()
            .ok_or_else(|| Error::custom("missing wire container"))?;

        self.current = frame.pop_front();
        let ended = self.current.is_none();
        container.advance::<Error>(ended)?;
        if ended {
            self.frames.pop();
        }

        Ok(())
    }

    async fn decode_any<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        match self.peek_kind().await? {
            Kind::Map => self.decode_map(visitor).await,
            Kind::Seq => self.decode_seq(visitor).await,
            Kind::Leaf => {
                let raw = self.take()?;
                Python::attach(|py| {
                    let raw = raw.bind(py);
                    if raw.is_none() {
                        visitor.visit_unit()
                    } else if raw.is_instance_of::<PyBool>() {
                        visitor.visit_bool(raw.extract()?)
                    } else if raw.is_instance_of::<PyInt>() {
                        if let Ok(n) = raw.extract::<i64>() {
                            visitor.visit_i64(n)
                        } else {
                            visitor.visit_u64(raw.extract::<u64>()?)
                        }
                    } else if raw.is_instance_of::<PyFloat>() {
                        visitor.visit_f64(raw.extract()?)
                    } else if let Ok(text) = raw.cast::<PyString>() {
                        let text = text.to_str()?;
                        self.budget.charge(text.len())?;
                        visitor.visit_string(text.to_owned())
                    } else {
                        Err(Error::custom("expected a primitive Python wire value"))
                    }
                })
            }
        }
    }

    async fn inspect_any<F>(&mut self, mut inspect: F) -> Result<(), Error>
    where
        F: for<'a> FnMut(Inspection<'a>) -> Result<(), Error> + Send,
    {
        let mut containers = Vec::new();
        loop {
            let kind = self.peek_kind().await?;
            if kind == Kind::Leaf {
                let raw = self.take()?;
                Python::attach(|py| -> Result<(), Error> {
                    let raw = raw.bind(py);
                    if let Ok(text) = raw.cast::<PyString>() {
                        let bytes = text.to_str()?.as_bytes();
                        inspect(Inspection::TextChunk(bytes))?;
                        inspect(Inspection::TextEnd {
                            capacity_bound: bytes.len(),
                        })?;
                    } else if !(raw.is_none()
                        || raw.is_instance_of::<PyBool>()
                        || raw.is_instance_of::<PyInt>()
                        || raw.is_instance_of::<PyFloat>())
                    {
                        return Err(Error::custom("expected primitive wire data"));
                    }

                    Ok(())
                })?;
            } else {
                let container = self.open_container(kind, None).await?;
                let empty = container.slot() == Slot::End;
                containers.push(container);
                if !empty {
                    continue;
                }
            }

            loop {
                if containers
                    .last()
                    .is_some_and(|container| container.slot() == Slot::End)
                {
                    let container = containers.pop().expect("completed inspection container");
                    inspect(if container.kind() == Kind::Map {
                        Inspection::Map {
                            len: container.len(),
                        }
                    } else {
                        Inspection::Sequence {
                            len: container.len(),
                            size_hint: None,
                        }
                    })?;
                } else if let Some(container) = containers.last_mut() {
                    self.finish_child(container).await?;
                    if container.slot() != Slot::End {
                        break;
                    }
                } else {
                    return Ok(());
                }
            }
        }
    }

    scalar_decoders!(decode_bool, decode_string, decode_unit);

    numeric_decoders!(decode_i8 => visit_i8(i8), decode_i16 => visit_i16(i16), decode_i32 => visit_i32(i32), decode_i64 => visit_i64(i64),
        decode_u8 => visit_u8(u8), decode_u16 => visit_u16(u16), decode_u32 => visit_u32(u32), decode_u64 => visit_u64(u64),
        decode_f32 => visit_f32(f32), decode_f64 => visit_f64(f64));

    async fn decode_bytes<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        self.decode_string(visitor).await
    }

    async fn decode_map<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        let container = self.open_container(Kind::Map, None).await?;
        visitor
            .visit_map(Access::<()> {
                decoder: self,
                container,
                marker: PhantomData,
            })
            .await
    }

    async fn decode_seq<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        let container = self.open_container(Kind::Seq, None).await?;
        visitor
            .visit_seq(Access::<()> {
                decoder: self,
                container,
                marker: PhantomData,
            })
            .await
    }

    async fn decode_tuple<V: Visitor>(
        &mut self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Error> {
        let container = self.open_container(Kind::Seq, Some(len)).await?;
        visitor
            .visit_seq(Access::<()> {
                decoder: self,
                container,
                marker: PhantomData,
            })
            .await
    }

    async fn decode_option<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        let none = Python::attach(|py| {
            self.current
                .as_ref()
                .is_some_and(|value| value.bind(py).is_none())
        });

        if none {
            self.take()?;
            visitor.visit_none()
        } else {
            visitor.visit_some(self).await
        }
    }

    async fn decode_uuid<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        self.decode_string(visitor).await
    }

    async fn decode_ignored_any<V: Visitor>(&mut self, visitor: V) -> Result<V::Value, Error> {
        self.inspect_any(|_| Ok(())).await?;
        visitor.visit_unit()
    }

    array_decoders!(decode_array_bool => visit_array_bool(bool), decode_array_i8 => visit_array_i8(i8),
        decode_array_i16 => visit_array_i16(i16), decode_array_i32 => visit_array_i32(i32), decode_array_i64 => visit_array_i64(i64),
        decode_array_u8 => visit_array_u8(u8), decode_array_u16 => visit_array_u16(u16), decode_array_u32 => visit_array_u32(u32), decode_array_u64 => visit_array_u64(u64),
        decode_array_f32 => visit_array_f32(f32), decode_array_f64 => visit_array_f64(f64));
}

struct Access<'a, T> {
    decoder: &'a mut Decoder,
    container: Container,
    marker: PhantomData<T>,
}

impl de::MapAccess for Access<'_, ()> {
    type Error = Error;

    async fn next_key<K: FromStream>(&mut self, context: K::Context) -> Result<Option<K>, Error> {
        if self.container.slot() == Slot::End {
            return Ok(None);
        }

        if self.container.slot() != Slot::Key {
            return Err(Error::custom("expected map value"));
        }

        let key = K::from_stream(context, self.decoder).await?;
        self.decoder.finish_child(&mut self.container).await?;
        Ok(Some(key))
    }

    async fn next_value<V: FromStream>(&mut self, context: V::Context) -> Result<V, Error> {
        if self.container.slot() != Slot::Value {
            return Err(Error::custom("expected map key"));
        }

        let value = V::from_stream(context, self.decoder).await?;
        self.decoder.finish_child(&mut self.container).await?;
        Ok(value)
    }
}

impl de::SeqAccess for Access<'_, ()> {
    type Error = Error;

    async fn next_element<T: FromStream>(
        &mut self,
        context: T::Context,
    ) -> Result<Option<T>, Error> {
        if self.container.slot() == Slot::End {
            return Ok(None);
        }

        let value = T::from_stream(context, self.decoder).await?;
        self.decoder.finish_child(&mut self.container).await?;
        Ok(Some(value))
    }
}

impl<T: FromStream<Context = ()> + Copy + Send> de::ArrayAccess<T> for Access<'_, T> {
    type Error = Error;

    async fn buffer(&mut self, buffer: &mut [T]) -> Result<usize, Error> {
        let mut count = 0;
        for slot in buffer {
            if self.container.slot() == Slot::End {
                break;
            }

            *slot = T::from_stream((), self.decoder).await?;
            self.decoder.finish_child(&mut self.container).await?;
            count += 1;
        }

        Ok(count)
    }
}
