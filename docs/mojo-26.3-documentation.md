# Mojo 官方文档完整提取

本文档包含从以下 Modular 官方页面提取的完整内容：
- https://docs.modular.com/mojo/lib/
- https://docs.modular.com/mojo/manual/

---

## 目录
1. [Mojo 标准库参考 (mojo/lib/)](#mojo-标准库参考-mojolib)
2. [Mojo 手册 (mojo/manual/)](#mojo-手册-mojomanual)

---

## Mojo 标准库参考 (mojo/lib/)

### 概述
This section includes the Mojo API references:
- **Standard library**: Common Mojo APIs.
- **MAX AI Kernels library**: Mojo APIs for writing high-performance computational kernels and custom operations for AI models.
- **Decorators**: Mojo decorators reference.

---

### 如何阅读 Mojo API 文档
Mojo syntax is covered in detail in the Mojo manual. Here's a quick cheat-sheet on reading struct and function signatures.

#### 参数 (Arguments)
Function arguments appear in parentheses after the function name:
```mojo
def example_fn(pos: Int, /, pos_or_kw: Int, *, kw_only: Bool = False):
    ...
```

Here's a quick overview of some special syntax in the argument list:

| 语法 | 说明 |
|------|------|
| **Slash (/)** | Arguments declared before a slash are positional-only arguments. |
| **Star (\*)** | A star by itself in place of an argument indicates that the arguments after the star are keyword-only. |
| **Equals sign (=)** | Introduces a default value for an optional argument. |

You may also see argument names prefixed with one or two stars (*):
```mojo
def myfunc2(*names, **attributes) raises:
    ...
```

- An argument name prefixed by a single star character, like `*names` identifies a variadic argument.
- An argument name prefixed with a double star, like `**attributes` identifies a variadic keyword-only argument.

An argument may also be preceded by an argument convention, which indicates how the value is passed:
```mojo
def sort(mut names: List[String]):
    ...
```

The most common conventions are:

| 约定 | 说明 |
|------|------|
| **read (default)** | The callee receives an immutable reference to the value. |
| **mut** | The callee receives a mutable reference to the value. |
| **owned** | The callee receives ownership of a value. |

For details and a complete list of argument conventions, see [Argument conventions](https://docs.modular.com/mojo/manual/values/argument-conventions/).

---

### 参数 (Parameters)
Mojo structs and functions can take parameters. Parameters are evaluated at compilation time, and act as constants at runtime. Parameter lists are enclosed in square brackets:
```mojo
struct ExampleStruct[size: Int, //, thing: Thing[size]]:
    ...
```

Parameters that occur before a double-slash (//) in the parameter list are infer-only parameters. You usually don't need to specify infer-only parameters; as the name suggests, they're usually inferred.

Like arguments, parameters can be positional-only, keyword-or-positional, or keyword-only, and they can be required or optional. The /, *, and = characters have the same meaning in parameter lists as they do in argument lists.

---

### 标准库 (Standard Library)
The Mojo standard library provides nearly everything you'll need for writing Mojo programs, including basic data types like `Int` and `SIMD`, collection types like `List`, reusable algorithms and modules to support GPU programming.

Top-level packages:

| 包名 | 说明 |
|------|------|
| 🗃️ **algorithm** | High performance data operations: vectorization, parallelization, reduction, memory. |
| 🗃️ **base64** | Binary data encoding: base64 and base16 encode/decode functions. |
| 🗃️ **benchmark** | Performance benchmarking: statistical analysis and detailed reports. |
| 🗃️ **bit** | Bitwise operations: manipulation, counting, rotation, and power-of-two utilities. |
| 🗃️ **builtin** | Language foundation: built-in types, traits, and fundamental operations. |
| 🗃️ **collections** | Core data structures: List, Dict, Set, Optional, plus specialized collections. |
| 🗃️ **compile** | Runtime function compilation and introspection: assembly, IR, linkage, metadata. |
| 🗃️ **complex** | Complex numbers: SIMD types, scalar types, and operations. |
| 🗃️ **documentation** | Documentation built-ins: decorators and utilities for doc generation. |
| 🗃️ **ffi** | Foreign function interface (FFI) for calling C code and loading libraries. |
| 🗃️ **format** | Provides formatting traits for converting types to text. |
| 🗃️ **gpu** | GPU programming primitives: thread blocks, async memory, barriers, and sync. |
| 🗃️ **hashlib** | Cryptographic and non-cryptographic hashing with customizable algorithms. |
| 🗃️ **io** | Core I/O operations: console input/output, file handling, writing traits. |
| 🗃️ **iter** | Iteration traits and utilities: Iterable, IterableOwned, Iterator, enumerate, zip, map. |
| 🗃️ **itertools** | Iterator tools for lazy sequence generation and transformation. |
| 🗃️ **logger** | Logging with configurable severity levels. |
| 🗃️ **math** | Math functions and constants: trig, exponential, logarithmic, and special functions. |
| 🗃️ **memory** | Low-level memory management: pointers, allocations, address spaces. |
| 🗃️ **os** | OS interface layer: environment, filesystem, process control. |
| 🗃️ **pathlib** | Filesystem path manipulation and navigation. |
| 📄️ **prelude** | Standard library prelude: fundamental types, traits, and operations auto-imported. |
| 🗃️ **pwd** | Password database lookups for user account information. |
| 🗃️ **python** | Python interoperability: import packages and modules, call functions, type conversion. |
| 🗃️ **random** | Pseudorandom number generation with uniform and normal distributions. |
| 🗃️ **reflection** | Compile-time reflection utilities for introspecting Mojo types and functions. |
| 🗃️ **runtime** | Runtime services: async execution and program tracing. |
| 🗃️ **stat** | File type constants and detection from stat system calls. |
| 🗃️ **subprocess** | Execute external processes and commands. |
| 🗃️ **sys** | System runtime: I/O, hardware info, intrinsics, compile-time utils. |
| 🗃️ **tempfile** | Manage temporary files and directories: create, locate, and cleanup. |
| 🗃️ **testing** | Unit testing: Assertions (equal, true, raises) and test suites. |
| 🗃️ **time** | Timing operations: monotonic clocks, performance counters, sleep, time_function. |
| 🗃️ **utils** | General utils: indexing, variants, static tuples, and thread synchronization. |

---

### MAX AI 内核库 (MAX AI Kernels library)
The MAX AI kernels library provides a collection of highly optimized, reusable compute kernels for high-performance numerical and AI workloads. These kernels serve as the foundational building blocks for writing MAX custom operations or standalone GPU kernels that are portable across CPUs and GPUs.

Top-level packages:

| 包名 | 说明 |
|------|------|
| 🗃️ **comm** | Provides communication primitives for GPUs. |
| 🗃️ **extensibility** | Includes the tensor package. |
| 🗃️ **kv_cache** | Contains implementations for several types of key-value caches. |
| 🗃️ **layout** | Provides layout and layout tensor types, which abstract memory layout for multidimensional data. |
| 🗃️ **linalg** | Provides CPU and GPU implementations of linear algebra functions. |
| 🗃️ **nn** | Provides neural network operators for deep learning models. |
| 🗃️ **nvml** | Implements wrappers around the NVIDIA Management Library (nvml). |
| 🗃️ **quantization** | This package contains a set of APIs for quantizing tensor data. |

---

### 装饰器 (Decorators)
A Mojo decorator is a higher-order function that modifies or extends the behavior of a struct, a function, or some other code.

| 装饰器 | 说明 |
|--------|------|
| 📄️ **@align** | Specifies a minimum alignment for a struct. |
| 📄️ **@always_inline** | Copies the body of a function directly into the body of the calling function. |
| 📄️ **@compiler.register** | Registers a custom operation for use with the MAX Graph API. |
| 📄️ **@__copy_capture** | Captures register-passable typed values by copy. |
| 📄️ **@deprecated** | Mojo's `@deprecated` decorator marks outdated APIs and schedules them for removal. When used with the `use` parameter, it also provides migration suggestions. |
| 📄️ **@doc_hidden** | Hides declarations from generated documentation. |
| 📄️ **@explicit_destroy** | Prevents automatic destruction by a `__del__()` method and requires explicit cleanup through named destructor methods. |
| 📄️ **@export** | Marks a function for export. |
| 📄️ **@fieldwise_init** | Generates fieldwise constructor for a struct. |
| 📄️ **@implicit** | Marks a constructor as eligible for implicit conversion. |
| 📄️ **@no_inline** | Prevents a function from being inlined. |
| 📄️ **@parameter** | Executes a function or if statement at compile time. |
| 📄️ **@staticmethod** | Declares a struct method as static. |

---

## Mojo 手册 (mojo/manual/)

### 欢迎
Welcome to the Mojo Manual, your complete guide to the Mojo🔥 programming language!

Combined with the Mojo API reference, this documentation provides everything you need to write high-performance Mojo code for CPUs and GPUs. If you see anything that can be improved, please file an issue or send a pull request for the docs on GitHub.

---

### 关于 Mojo (About Mojo)
Mojo is a systems programming language specifically designed for high-performance AI infrastructure and heterogeneous hardware. Its Pythonic syntax makes it easy for Python programmers to learn and it fully integrates the existing Python ecosystem, including its wealth of AI and machine-learning libraries.

It's the first programming language built from the ground-up using MLIR—a modern compiler infrastructure for heterogeneous hardware, from CPUs to GPUs and other AI ASICs. That means you can use one language to write all your code, from high-level AI applications all the way down to low-level GPU kernels, without using any hardware-specific libraries (such as CUDA and ROCm).

Learn more about it in the [Mojo vision doc](https://docs.modular.com/mojo/vision/).

---

### 主要特性 (Key features)

| 特性 | 说明 |
|------|------|
| **Python syntax & interop** | Mojo adopts (and extends) Python's syntax and integrates with existing Python code. Mojo's interoperability works in both directions, so you can import Python libraries into Mojo and create Mojo bindings to call from Python. Read about [Python interop](https://docs.modular.com/mojo/manual/python/). |
| **Struct-based types** | All data types—including basic types such as String and Int—are defined as structs. No types are built into the language itself. That means you can define your own types that have all the same capabilities as the standard library types. Read about [structs](https://docs.modular.com/mojo/manual/types/structs/). |
| **Zero-cost traits** | Mojo's trait system solves the problem of static typing by letting you define a shared set of behaviors that types (structs) can implement. It allows you to write functions that depend on traits rather than specific types, similar to interfaces in Java or protocols in Swift, except with compile-time type checking and no run-time performance cost. Read about [traits](https://docs.modular.com/mojo/manual/types/traits/). |
| **Value semantics** | Mojo supports both value and reference semantics, but generally defaults to value semantics. With value semantics, each copy is independent—modifying one copy won't affect another. With reference semantics, multiple variables can point to the same instance (sometimes called an object), so changes made through one variable are visible through all others. Mojo-native types predominantly use value semantics, which prevents multiple variables from unexpectedly sharing the same data. Read about [value semantics](https://docs.modular.com/mojo/manual/values/value-semantics/). |
| **Value ownership** | Mojo's ownership system ensures that only one variable "owns" a specific value at a given time—such that Mojo can safely deallocate the value when the owner's lifetime ends—while still allowing you to share references to the value. This provides safety from errors such as use-after-free, double-free, and memory leaks without the overhead cost of a garbage collector. Read about [ownership](https://docs.modular.com/mojo/manual/values/ownership/). |
| **Compile-time metaprogramming** | Mojo's parameterization system enables powerful metaprogramming in which the compiler generates a unique version of a type or function based on parameter values, similar to C++ templates, but more intuitive. Read about [parameterization](https://docs.modular.com/mojo/manual/parameters/). |
| **Hardware portability** | Mojo is designed from the ground up to support heterogeneous hardware—the Mojo compiler makes no assumptions about whether your code is written for CPUs, GPUs, or something else. Instead, hardware behaviors are handled by Mojo libraries, as demonstrated by types such as SIMD that allows you to write vectorized code for CPUs, and the gpu package that enables hardware-agnostic GPU programming. Read about [GPU programming](https://docs.modular.com/mojo/manual/gpu/). |

---

### 开始使用 (Get started)

[Get started with a tutorial](https://docs.modular.com/mojo/manual/get-started/hello-world.html)

---

### 更多资源 (More resources)

(更多内容请访问 https://docs.modular.com/mojo/manual/)

---

## 文档提取说明

本文档使用 WebFetch 工具从以下 Modular 官方页面提取：
- https://docs.modular.com/mojo/lib/
- https://docs.modular.com/mojo/manual/

提取日期：2026-04-16

所有内容均保持原始格式和结构，以确保信息的准确性和完整性。
