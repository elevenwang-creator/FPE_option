# Mojo Language — New Features & Performance Advantages

> **Latest stable: v0.26.2 (March 19, 2026) · Nightly: v0.26.3**
> Built on MLIR · Python superset syntax · Systems-level performance

---

## Table of Contents

1. [Why Mojo?](#1-why-mojo)
2. [Performance Benchmarks](#2-performance-benchmarks)
3. [Language Fundamentals (Modern Syntax)](#3-language-fundamentals-modern-syntax)
4. [Compile-Time Metaprogramming (`comptime`)](#4-compile-time-metaprogramming-comptime)
5. [Ownership & Memory Model](#5-ownership--memory-model)
6. [SIMD-First Architecture](#6-simd-first-architecture)
7. [GPU Programming (No CUDA Needed)](#7-gpu-programming-no-cuda-needed)
8. [Python Interoperability](#8-python-interoperability)
9. [Advanced Type System](#9-advanced-type-system)
10. [Reflection & Introspection](#10-reflection--introspection)
11. [Standard Library Highlights](#11-standard-library-highlights)
12. [New in v0.26.1 – v0.26.3](#12-new-in-v0261--v0263)
13. [Mojo vs. Others (Comparison)](#13-mojo-vs-others-comparison)
14. [When to Use Mojo](#14-when-to-use-mojo)

---

## 1. Why Mojo?

Mojo is the first language **built from scratch on MLIR** (Multi-Level Intermediate Representation) — the same compiler infrastructure powering TensorFlow and PyTorch. It combines Python's readability with C++/Rust-level performance.

**Core philosophy:**

| Principle | Description |
|---|---|
| **Python superset** | Familiar syntax — Python devs learn it in hours |
| **Zero-cost abstractions** | Structs with no object overhead, no GC |
| **Heterogeneous compute** | Single language for CPU + GPU + TPU |
| **MLIR-native** | Compiles directly to optimized machine code |
| **Gradual adoption** | Full Python interop — migrate hot paths incrementally |

```
Python:                          Mojo:
-------                          -----
Source Code                      Source Code
    |                                |
    v                                v
Bytecode                         MLIR IR
    |                                |
    v                          +-----+-----+
Interpreter                    |     |     |
    |                         CPU   GPU   TPU
    v                         Code  Code  Code
Runtime Dispatch
```

---

## 2. Performance Benchmarks

Real-world benchmarks from the 2026 Optimization Ladder study:

### N-body Simulation (500K iterations)

| Approach | Time | Speedup | Notes |
|---|---|---|---|
| CPython 3.14 | 1,242ms | 1.0× | Baseline |
| PyPy | 98ms | 13× | JIT, limited ecosystem |
| Numba `@njit` | 22ms | 56× | Decorator-based |
| **Mojo** | **16ms** | **78×** | Native compilation |
| Cython | 10ms | 124× | Requires C knowledge |
| Rust PyO3 | 11ms | 113× | Requires learning Rust |

### Spectral-Norm (N=2000, Matrix Operations)

| Approach | Time | Speedup | Notes |
|---|---|---|---|
| CPython 3.14 | 14,046ms | 1.0× | Baseline |
| **Mojo** | **118ms** | **119×** | Native compilation |
| Rust PyO3 | 91ms | 154× | Requires learning Rust |
| NumPy (BLAS) | 27ms | 520× | Delegates to Fortran |

**Key takeaway:** Mojo achieves **78–119× speedup** over CPython, competing directly with Rust and Cython — but with Python-like syntax.

### Why Mojo Is Fast

1. **MLIR-native** — compiles to hardware, no interpretation
2. **SIMD built-in** — vectorization is a first-class primitive
3. **No GC** — ownership model eliminates garbage collection pauses
4. **Zero object overhead** — a Mojo `Int` is exactly 8 bytes (Python: 28 bytes)
5. **Compile-time metaprogramming** — loops and branches resolved at compile time

---

## 3. Language Fundamentals (Modern Syntax)

### `def` is the only function keyword

`fn` is deprecated. `def` does **not** implicitly raise — add `raises` explicitly:

```mojo
def compute(x: Int) -> Int:              # non-raising (compiler enforced)
    return x * 2

def load(path: String) raises -> String: # explicitly raising
    return open(path).read()

def main() raises:
    print(compute(42))
```

### Variable declarations

```mojo
var x = 42              # mutable variable (no `let` keyword)
var name: String = "Mojo"
```

### `comptime` replaces `alias`

```mojo
comptime N = 1024                           # compile-time constant
comptime MyType = Int                       # type alias
comptime if N > 512:                        # compile-time branch
    print("Large")
comptime for i in range(10):               # compile-time loop (fully unrolled)
    print(i)
comptime assert N > 0, "N must be positive" # compile-time assertion
```

### Argument conventions

| Convention | Meaning | Example |
|---|---|---|
| *(implicit)* | `read` — immutable borrow | `def show(self):` |
| `mut` | mutable reference | `def modify(mut self):` |
| `var` | owned (can be consumed) | `def take(var value: String):` |
| `out` | uninitialized output | `def __init__(out self):` |
| `deinit` | consuming/destroying | `def consume(deinit self):` |
| `ref` | reference with origin | `def view(ref self):` |

### Struct patterns

```mojo
@fieldwise_init
struct Point(Copyable, Movable, Writable):
    var x: Float64
    var y: Float64
    # Compiler auto-generates __init__, copy/move constructors, write_to()

# Trait composition with &
struct Node[T: Copyable & Writable]:
    var value: Self.T          # Self-qualify struct parameters (mandatory)
```

### Collection literals

```mojo
var nums = [1, 2, 3]                         # List[Int]
var nums: List[Float32] = [1.0, 2.0, 3.0]    # explicit element type
var scores = {"alice": 95, "bob": 87}         # Dict[String, Int]
```

### T-Strings (template strings) — NEW in v0.26.2

```mojo
var x = 10
var y = 20
print(t"{x} + {y} = {x + y}")  # 10 + 20 = 30
# Produces TString — lazy, type-safe, no immediate allocation
```

### Assert statement — NEW in v0.26.2

```mojo
assert x > 0, "x must be positive"
assert len(items) != 0
# Desugars to debug_assert(), respects -D ASSERT flag
```

---

## 4. Compile-Time Metaprogramming (`comptime`)

Mojo's compile-time system is among the most powerful of any language:

```mojo
# Constants and type aliases
comptime N = 1024
comptime ElementType = Float32

# Compile-time conditionals — branches removed at compile time
comptime if has_accelerator():
    print("GPU available")

# Compile-time loops — fully unrolled
comptime for i in range(8):
    process_lane(i)

# Compile-time assertions (inside function bodies)
comptime assert N > 0, "N must be positive"

# Force compile-time evaluation of expressions (v0.26.1+)
print(comptime(some_layout.size()))
```

### Struct-level `comptime`

```mojo
struct Config:
    comptime DefaultSize = 64
    comptime ElementType = Float32
    comptime MAX_ITEMS = 1024
```

---

## 5. Ownership & Memory Model

Mojo uses a **Rust-inspired ownership model** — no garbage collector, no runtime overhead.

### Lifecycle methods (modern syntax)

```mojo
struct MyBuffer:
    var data: UnsafePointer[UInt8, MutExternalOrigin]

    # Constructor
    def __init__(out self, size: Int):
        self.data = alloc[UInt8](size)

    # Copy constructor (keyword-only `copy` arg)
    def __init__(out self, *, copy: Self):
        self.data = alloc[UInt8](copy.size)
        memcpy(self.data, copy.data, copy.size)

    # Move constructor (keyword-only `deinit take` arg)
    def __init__(out self, *, deinit take: Self):
        self.data = take.data

    # Destructor
    def __del__(deinit self):
        self.data.free()
```

### Explicit copy / transfer

```mojo
var a = MyBuffer(1024)
var b = a.copy()    # explicit copy (Copyable trait)
var c = a^          # ownership transfer (a is consumed)
```

### Origin system (not "lifetimes")

Mojo tracks reference provenance with **origins**:

```mojo
struct Span[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]]:
    ...

# Key origin types:
# Origin, MutOrigin, ImmutOrigin, MutAnyOrigin, ImmutAnyOrigin
# MutExternalOrigin, StaticConstantOrigin
# Use origin_of(value) to get a value's origin
```

### Explicitly-destroyed types (linear types) — NEW in v0.26.1

Types without `__del__()` force the programmer to explicitly handle cleanup:

```mojo
@explicit_destroy
struct DatabaseConnection:
    def close(deinit self):
        # Must be called explicitly — compiler enforces this
        ...
```

### Pointer types

| Type | Use |
|---|---|
| `Pointer[T]` | Safe, non-nullable. Deref with `p[]` |
| `UnsafePointer[T]` | Raw pointer, `.free()` required |
| `OwnedPointer[T]` | Unique ownership (like Rust `Box`) |
| `ArcPointer[T]` | Reference-counted shared ownership |
| `Span[T]` | Non-owning contiguous view |

---

## 6. SIMD-First Architecture

SIMD is a **first-class citizen** in Mojo — not a library bolt-on:

```mojo
# Construction
var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

# Lane access
v[0]                              # read lane
v[0] = 5.0                        # write lane

# Type cast
v.cast[DType.uint32]()            # element-wise cast

# Clamp
v.clamp(0.0, 1.0)                # element-wise clamp

# min/max — free functions
from std.math import min, max
min(a, b)                          # element-wise min
max(a, b)                          # element-wise max

# Boolean masks and selection
var mask = (v > 0.0)              # SIMD[DType.bool, 4]
mask.select(true_val, false_val)  # per-lane ternary

# Reductions
v.reduce_add()                     # horizontal sum → Scalar
v.reduce_max()                     # horizontal max
v.reduce_min()                     # horizontal min
```

### Vectorize pattern

```mojo
from std.algorithm import vectorize

def square_elements(ptr: UnsafePointer[Float32], size: Int):
    def inner[width: Int](i: Int):
        var val = ptr.load[width=width](i)
        ptr.store[width=width](i, val * val)

    vectorize[simd_width_of[DType.float32]()](size, inner)
```

### 128-bit and 256-bit integers (v25.2+)

```mojo
var big: Int128 = 170141183460469231731687303715884105727
var huge: UInt256 = ...
# DType.int128, DType.uint128, DType.int256, DType.uint256
```

---

## 7. GPU Programming (No CUDA Needed)

Mojo provides **native GPU programming** — no CUDA, no `__global__`, no `<<<>>>`.

### Key concept mapping

| CUDA | Mojo GPU |
|---|---|
| `__global__ void kernel(...)` | Plain `def kernel(...)` |
| `kernel<<<grid, block>>>(args)` | `ctx.enqueue_function[kernel, kernel](args, grid_dim=..., block_dim=...)` |
| `cudaMalloc` | `ctx.enqueue_create_buffer[dtype](count)` |
| `cudaMemcpy` | `ctx.enqueue_copy(dst, src)` |
| `__syncthreads()` | `barrier()` |
| `__shared__ float s[N]` | `LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()` |
| `threadIdx.x` | `thread_idx.x` (returns `Int` as of nightly v0.26.3) |

### Complete GPU example — Vector Addition

```mojo
from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime dtype = DType.float32
comptime N = 1024
comptime BLOCK = 256
comptime layout = Layout.row_major(N)

def add_kernel(
    a: LayoutTensor[dtype, layout, MutAnyOrigin],
    b: LayoutTensor[dtype, layout, MutAnyOrigin],
    c: LayoutTensor[dtype, layout, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        c[tid] = a[tid] + b[tid]

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)

    var a = LayoutTensor[dtype, layout](a_buf)
    var b = LayoutTensor[dtype, layout](b_buf)
    var c = LayoutTensor[dtype, layout](c_buf)

    ctx.enqueue_function[add_kernel, add_kernel](
        a, b, c, N,
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )

    with c_buf.map_to_host() as host:
        var result = LayoutTensor[dtype, layout](host)
        print(result)  # All 3.0
```

### LayoutTensor — Primary GPU data abstraction

```mojo
# Create layouts
comptime layout_1d = Layout.row_major(1024)
comptime layout_2d = Layout.row_major(64, 64)

# Tiling for shared memory optimization
var tile = tensor.tile[TILE_M, TILE_K](Int(block_idx.y), Int(block_idx.x))

# Shared memory allocation
var shared = LayoutTensor[dtype, tile_layout, MutAnyOrigin,
    address_space=AddressSpace.SHARED].stack_allocation()
```

### Supported hardware

- **NVIDIA**: SM50+ (Tensor cores on SM70+, TMA on SM90+ Hopper)
- **AMD**: CDNA (MI250X+), RDNA3+
- **Apple Silicon**: M-series GPUs (M1–M5, including M5 detection in nightly)

---

## 8. Python Interoperability

### Calling Python from Mojo

```mojo
from std.python import Python, PythonObject

def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([1, 2, 3])
    print(arr.shape)

    # PythonObject → Mojo: MUST use `py=` keyword
    var i = Int(py=some_py_obj)
    var f = Float64(py=some_py_obj)
    var s = String(py=some_py_obj)
```

### Calling Mojo from Python (Extension modules)

```mojo
from std.python.bindings import PythonModuleBuilder

@export
fn PyInit_my_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("my_module")
        m.def_function[add]("add")
        return m.finalize()
    except e:
        abort(String("failed: ", e))

fn add(a: PythonObject, b: PythonObject) raises -> PythonObject:
    return a + b
```

```python
# From Python:
import mojo.importer       # enables auto-compile of .mojo files
import my_module
print(my_module.add(1, 2))  # 3
```

### Exposing Mojo types to Python

```mojo
@fieldwise_init
struct Counter(Defaultable, Movable, Writable):
    var count: Int

    @staticmethod
    fn increment(py_self: PythonObject) raises -> PythonObject:
        var ptr = py_self.downcast_value_ptr[Self]()
        ptr[].count += 1
        return PythonObject(ptr[].count)
```

---

## 9. Advanced Type System

### Conditional trait conformances — NEW in v0.26.2

```mojo
struct List[T: Movable](
    Copyable,
    Equatable where conforms_to(T, Equatable),   # only when T is Equatable
    Movable,
    Writable where conforms_to(T, Writable),     # only when T is Writable
):
    ...
```

### Traits with default implementations — NEW in v0.26.1

Simple structs get `Hashable`, `Writable`, `Equatable` for free:

```mojo
@fieldwise_init
struct Point(Hashable, Writable, Equatable):
    var x: Float64
    var y: Float64
    # All trait methods auto-derived from fields — zero boilerplate!

print(Point(1.5, 2.7))              # Point(x=1.5, y=2.7)
print(Point(1, 2) == Point(1, 2))   # True
hash(Point(3.0, 4.0))               # works automatically
```

### Typed errors — NEW in v0.26.1

```mojo
def parse(s: String) raises Int -> Float64:
    raise 42  # raises an Int, not Error

try:
    var result = parse("bad")
except err:    # err is typed as Int
    print("error code:", err)
```

Typed errors are **zero-cost** — they compile to an alternate return value with no stack unwinding. Works on GPU and embedded targets.

### Never type — NEW in v0.26.1

```mojo
def abort_always() raises Never:
    raise Error("fatal")
    # Function provably never returns normally
```

### `@align(N)` decorator — NEW in v0.26.2

```mojo
@align(64)
struct CacheAligned:
    var data: Int
# sizeof alignment = 64 bytes (cache-line aligned)

@align(Self.alignment)
struct AlignedBuffer[alignment: Int]:
    var data: Int
# Parameterized alignment!
```

### Type hierarchy

```
AnyType
  ImplicitlyDestructible          — auto __del__; most types
  Movable                         — move constructor
    Copyable                      — copy constructor
      ImplicitlyCopyable
    RegisterPassable
      TrivialRegisterPassable     — fits in registers, trivial copy
```

### Common decorators

| Decorator | Purpose |
|---|---|
| `@fieldwise_init` | Generate fieldwise constructor |
| `@implicit` | Allow implicit conversion |
| `@always_inline` | Force inline |
| `@no_inline` | Prevent inline |
| `@staticmethod` | Static method |
| `@deprecated("msg")` | Deprecation warning |
| `@explicit_destroy` | Linear type (no implicit destruction) |
| `@align(N)` | Minimum alignment (v0.26.2+) |

---

## 10. Reflection & Introspection

Massively expanded in v0.26.1:

```mojo
from reflection import (
    struct_field_count, struct_field_names, struct_field_types,
    offset_of, source_location, call_location,
)

@fieldwise_init
struct Point:
    var x: Int
    var y: Float64

def main() raises:
    # Field introspection
    comptime count = struct_field_count[Point]()          # 2
    comptime names = struct_field_names[Point]()          # ["x", "y"]
    comptime x_offset = offset_of[Point, name="x"]()     # 0
    comptime y_offset = offset_of[Point, name="y"]()     # 8

    # Source location
    var loc = source_location()
    print(loc)  # main.mojo:15:15

    # Trait conformance checking on dynamic types
    comptime for i in range(struct_field_count[Point]()):
        comptime field_type = struct_field_types[Point]()[i]
        comptime if conforms_to(field_type, Writable):
            print("Field", i, "is Writable")
```

### Use cases

- **Automatic serialization/deserialization**
- **Debug formatting** (auto-derived `Writable`)
- **ORM-style field mapping**
- **Generic algorithms** operating on arbitrary struct fields

---

## 11. Standard Library Highlights

### Collections

| Type | Key Feature |
|---|---|
| `List[T]` | Dynamic array, bracket literal `[1, 2, 3]` |
| `Dict[K, V]` | Swiss Table implementation (v0.26.2), SIMD group probing |
| `Set[T]` | Hash set with conditional `Comparable` |
| `Deque[T]` | Double-ended queue |
| `InlineArray[T, N]` | Stack-allocated fixed array |
| `LinkedList[T]` | Doubly-linked list |
| `Optional[T]` | Nullable container |
| `Variant[*Ts]` | Type-safe tagged union |

### Dict Swiss Table upgrade (v0.26.2)

- **SIMD group probing** for lookups
- **7/8 load factor** (up from 2/3)
- In-place tombstone rehashing
- Custom `DictKeyError` for efficient error handling

### Strings

```mojo
var s = "Hello, Mojo! 🔥"
len(s)                      # byte length (UTF-8)
s.count_codepoints()        # codepoint count
s[byte=0]                   # byte-level access → StringSlice

# UTF-8 safe construction (v0.26.1+)
var safe   = String(from_utf8=raw_bytes)          # raises on invalid
var lossy  = String(from_utf8_lossy=raw_bytes)    # replaces invalid with �
var unsafe = String(unsafe_from_utf8=raw_bytes)   # no validation

# Static strings — zero allocation
comptime GREETING: StaticString = "Hello, World"

# T-strings — lazy interpolation (v0.26.2+)
var msg = t"Result: {compute(x)}"
```

### Iterators (v0.26.1+)

```mojo
# StopIteration protocol (Python-style)
struct MyIter:
    comptime Element: Movable = Int
    def __next__(mut self) raises StopIteration -> Self.Element:
        if self.done:
            raise StopIteration()
        return self.current_value

# Peekable iterators
from std.iter import peekable
var it = peekable(my_collection.__iter__())
print(it.peek())  # look ahead without advancing
```

### Testing

```mojo
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

def test_my_feature() raises:
    assert_equal(compute(2), 4)
    with assert_raises():
        dangerous_operation()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

### Benchmarking

```mojo
from std.benchmark import Bench, BenchConfig, Bencher, BenchId

@parameter
@always_inline
def bench_fn(mut b: Bencher) capturing raises:
    @parameter
    @always_inline
    def launch(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel, kernel](args, grid_dim=G, block_dim=B)
    b.iter_custom[launch](ctx)

var bench = Bench(BenchConfig(max_iters=50000))
bench.bench_function[bench_fn](BenchId("my_kernel"))
```

---

## 12. New in v0.26.1 – v0.26.3

### v0.26.1 (January 29, 2026)

| Feature | Description |
|---|---|
| **Reflection module** | `struct_field_count`, `struct_field_names`, `struct_field_types`, `offset_of`, `source_location` |
| **Typed errors** | `raises CustomError` — zero-cost, works on GPU |
| **`Never` type** | For functions that provably never return |
| **Explicitly-destroyed types** | Linear types with `@explicit_destroy` |
| **Trait defaults** | Auto-derived `Hashable`, `Writable`, `Equatable` |
| **`comptime(expr)`** | Force subexpression evaluation at compile time |
| **`...` expression** | `EllipsisType` for overloaded subscripts |
| **`StopIteration` iterators** | Python-style iterator protocol |
| **String UTF-8 safety** | `from_utf8=`, `from_utf8_lossy=`, `unsafe_from_utf8=` |
| **`DictKeyError`** | Custom error type for dict lookups |
| **Native file I/O** | Direct `libc` system calls, no CompilerRT |
| **`os.process`** | Spawn processes via `posix_spawn()` |
| **`Copyable` refines `Movable`** | No more redundant trait requirements |

### v0.26.2 (March 19, 2026)

| Feature | Description |
|---|---|
| **Conditional conformances** | `where` clauses on struct trait conformance |
| **`def`/`fn` unification** | `def` is non-raising by default; `fn` deprecated |
| **T-strings** | `t"..."` template strings with lazy evaluation |
| **`comptime if/for`** | Replaces `@parameter if/for` |
| **`assert` statement** | Runtime assertion (respects `-D ASSERT` flag) |
| **`@align(N)`** | Struct alignment control |
| **Init unification** | `__moveinit__` → `__init__(*, deinit take: Self)` |
| **Dict Swiss Table** | SIMD group probing, 7/8 load factor |
| **Traits no longer auto-`ImplicitlyDestructible`** | Encourages linear type ecosystem support |
| **Official AI agent skills** | `mojo-syntax`, `mojo-gpu-fundamentals`, `mojo-python-interop`, `new-modular-project` |

### v0.26.3 Nightly (in progress)

| Feature | Description |
|---|---|
| **Unicode escapes** | `\uXXXX` and `\UXXXXXXXX` in string literals |
| **Variadic pack forwarding** | `*pack` forwarding through runtime calls |
| **GPU `Int` migration** | `thread_idx` etc. return `Int` instead of `UInt` |
| **`IterableOwned` trait** | Consuming iteration over collections |
| **AMD MI250X support** | New GPU target |
| **Apple M5 detection** | `CompilationTarget.is_apple_m5()` |
| **Conditional `RegisterPassable`** | Per-parameter conformance |

---

## 13. Mojo vs. Others (Comparison)

| Feature | Mojo | Python | Rust | C++ |
|---|---|---|---|---|
| **Syntax** | Pythonic | Python | Unique | Complex |
| **Performance** | 78–119× Python | 1× | ~113–154× Python | ~100–200× Python |
| **Memory safety** | Ownership + origins | GC | Ownership + lifetimes | Manual / smart ptrs |
| **GPU programming** | Native, single language | CUDA/C++ required | wgpu/cuda crates | CUDA, HIP |
| **SIMD** | First-class built-in | NumPy/Numba | std::simd (nightly) | Intrinsics |
| **Python interop** | Native, bidirectional | — | PyO3 | pybind11 |
| **Compile-time** | `comptime` (very powerful) | None | `const fn` / macros | `constexpr` / templates |
| **Learning curve** | Low (Python devs) | Very low | Steep | Steep |
| **Ecosystem** | Growing (2026) | Massive | Large | Massive |
| **GC pauses** | None | Yes | None | None (if careful) |

---

## 14. When to Use Mojo

### ✅ Best for

- **Numerical computing** — tight loops, matrix ops, scientific simulation
- **GPU kernel development** — single language for CPU+GPU, no CUDA boilerplate
- **AI/ML infrastructure** — MLIR-native, tensor operations
- **Performance-critical Python hot paths** — incremental migration
- **Systems programming with Python ergonomics** — ownership without Rust's ceremony

### ⚠️ Consider alternatives when

- **Massive ecosystem needed** — Python's PyPI is unmatched
- **Production stability critical** — Mojo is still evolving rapidly
- **Web/mobile development** — not Mojo's focus
- **Existing Rust/C++ codebase** — switching cost may not justify

### Migration strategy

```
1. Profile Python code → identify hot paths
2. Extract hot path to .mojo file
3. Call Mojo from Python (import mojo.importer)
4. Gradually expand Mojo coverage
5. Full Mojo for new performance-critical modules
```

---

## Quick Reference Card

```mojo
# Variable
var x = 42

# Function (non-raising)
def add(a: Int, b: Int) -> Int:
    return a + b

# Function (raising)
def load(path: String) raises -> String:
    return open(path).read()

# Compile-time constant
comptime N = 1024

# Struct with auto-derived traits
@fieldwise_init
struct Vec3(Copyable, Movable, Writable, Equatable, Hashable):
    var x: Float64
    var y: Float64
    var z: Float64

# Collection literals
var items = [1, 2, 3]
var config = {"debug": True, "level": 5}

# SIMD
var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var sum = v.reduce_add()

# Template string
var msg = t"sum = {sum}"

# Assertion
assert sum > 0.0, "sum must be positive"

# Python interop
from std.python import Python
var np = Python.import_module("numpy")

# GPU (requires hardware)
from std.gpu import global_idx
from std.gpu.host import DeviceContext
```

---

*Generated from Mojo v0.26.2 documentation, changelog, and AI agent skills. Last updated: March 2026.*
