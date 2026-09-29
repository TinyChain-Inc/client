"""Declarations of Chain-owned persistent collection members."""

from .collection.btree import BTree
from .collection.table import Table
from .state import GetOpRef
from .state.base import State
from .uri import URI


class Sync:
    """Declare a SyncChain over an empty BTree or Table schema."""

    __uri__ = URI(State, "chain", "sync")

    def __init__(self, collection):
        if type(collection) not in (BTree, Table):
            raise TypeError("Sync requires a BTree or Table declaration")
        self.collection = collection
        collection._declaration()

    def declaration(self):
        return GetOpRef(str(self.__uri__), self.collection._declaration())

    def _bind(self, path):
        form = {path: []} if path.startswith("$") else GetOpRef(path, None)
        return type(self.collection)(form)
