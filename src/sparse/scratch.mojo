"""RAII scratch buffer for temporary workspace allocations."""

from std.memory import alloc


struct ScratchBuffer[T: ImplicitlyCopyable](Copyable, Movable):
    """RAII wrapper around alloc/free for temporary workspace buffers.

    Ensures memory is freed on scope exit, even if an exception is thrown.
    """

    var data: UnsafePointer[Self.T, MutExternalOrigin]
    var size: Int

    def __init__(out self, size: Int):
        self.data = alloc[Self.T](size)
        self.size = size

    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.data = alloc[Self.T](self.size)
        for i in range(self.size):
            self.data[i] = copy.data[i]

    def __init__(out self, *, deinit take: Self):
        self.data = take.data
        self.size = take.size

    def __del__(deinit self):
        if self.size > 0:
            self.data.free()

    @always_inline
    def __getitem__(self, i: Int) -> Self.T:
        return self.data[i]

    @always_inline
    def __setitem__(mut self, i: Int, val: Self.T):
        self.data[i] = val

    @always_inline
    def ptr(self) -> UnsafePointer[Self.T, MutExternalOrigin]:
        return self.data

    @always_inline
    def __len__(self) -> Int:
        return self.size
