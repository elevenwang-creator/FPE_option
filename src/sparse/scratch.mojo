"""RAII scratch buffer for temporary workspace allocations.

Ensures heap memory allocated via `alloc[]` is freed on scope exit,
even through moves and exceptions.

Uses `UnsafePointer[type, MutExternalOrigin]` which is the standard
pattern for manually-managed heap memory in Mojo. The `alloc()` function
returns memory with `MutExternalOrigin` that must be explicitly freed
via `.free()` — this struct automates that contract.

Memory functions used:
  - `alloc` / `.free()` — heap allocation / deallocation
  - `uninit_copy_n` — copy-initializes buffer (uses SIMD `memcpy`
    for trivial types, element-by-element `init_pointee_copy` otherwise)
  - `memset_zero` — fast SIMD zero-fill

See: https://mojolang.org/docs/manual/pointers/unsafe-pointers/
     https://mojolang.org/docs/std/memory/memory/
"""

from std.memory import alloc, memset_zero, uninit_copy_n


struct ScratchBuffer[T: ImplicitlyCopyable](Movable):
    var data: UnsafePointer[Self.T, MutExternalOrigin]
    var size: Int

    def __init__(out self, size: Int):
        self.data = alloc[Self.T](size)
        self.size = size

    def __init__(out self, *, deinit existing: Self):
        self.data = existing.data
        self.size = existing.size

    def __del__(deinit self):
        if self.size > 0:
            self.data.free()

    @always_inline
    def fill_zero(mut self):
        memset_zero(self.data, self.size)

    @always_inline
    def __getitem__(self, i: Int) -> Self.T:
        return self.data[i]

    @always_inline
    def __setitem__(mut self, i: Int, val: Self.T):
        (self.data + i).init_pointee_copy(val)

    @always_inline
    def ptr(self) -> UnsafePointer[Self.T, MutExternalOrigin]:
        return self.data

    @always_inline
    def __len__(self) -> Int:
        return self.size
