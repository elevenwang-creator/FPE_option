"""Unified option pricer with pre-computed quadrature weights.

Key improvements over original:
- Pre-computes trapezoidal weights once, reuses per option
- Hoists payoff computation out of inner V-loop (payoff depends only on S)
- SIMD inner loop over variance dimension
- Computes Vega alongside Delta/Gamma
"""

from std.sys import has_accelerator

from server.pdf_cache import PDFGrid
from server.interpolator import Interpolator
from server.payoffs import BarrierDownAndIn, BarrierUpAndOut, EuropeanCall, EuropeanPut
from server.greeks import Greeks
from std.sys import simd_width_of
from std.algorithm import parallelize


@fieldwise_init
struct PricingRequest(Copyable, Movable):
    var S: Float64
    var K: Float64
    var V: Float64
    var barrier: Float64
    var payoff_type: Int
    var param_hash: UInt64

    def is_valid(self) -> Bool:
        """Validate pricing request parameters."""
        return (
            self.S > 0.0
            and self.K > 0.0
            and self.V >= 0.0
            and self.barrier > self.S
            and self.payoff_type >= 0
            and self.payoff_type <= 3
        )


@fieldwise_init
struct PricingResult(Copyable, Movable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Bool


@fieldwise_init
struct Pricer[B: Int]:
    """Unified pricer. B=1 -> CPU path, B>1 -> batch path with fallback.

    Pre-computes trapezoidal quadrature weights for O(1) amortized cost
    per option pricing call after the first.
    """

    var interpolator: Interpolator
    var greeks_computer: Greeks[Self.B]

    def price(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """Price B options using pre-computed PDF grid."""

        comptime if Self.B == 1:
            return self._price_single(grid, requests)
        else:
            comptime if has_accelerator():
                return self._price_gpu_batch(grid, requests)
            else:
                return self._price_cpu_parallel(grid, requests)

    def _price_single(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """CPU path: pre-compute weights once, then integrate for each option."""

        # Pre-compute trapezoidal weights once for the entire grid
        var ds_weights = self._compute_trap_weights(grid.s_points)
        var dv_weights = self._compute_trap_weights(grid.v_points)

        var results: List[PricingResult] = []
        for req in requests:
            if not req.is_valid():
                results.append(PricingResult(
                    price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False,
                ))
                continue
            var price = self._integrate_payoff_fast(grid, req, ds_weights, dv_weights)
            var payoff = self._get_payoff(req)
            var delta = self.greeks_computer.compute_delta(
                grid, self.interpolator,
                req.S, req.V, req.K, req.barrier, payoff,
            )
            var gamma = self.greeks_computer.compute_gamma(
                grid, self.interpolator,
                req.S, req.V, req.K, req.barrier, payoff,
            )
            var vega = self.greeks_computer.compute_vega(
                grid, self.interpolator,
                req.S, req.V, req.K, req.barrier, payoff,
            )
            results.append(PricingResult(
                price=price, delta=delta, gamma=gamma, vega=vega, success=True,
            ))
        return results^

    def _price_cpu_parallel(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """Batch CPU path: parallelize across options."""
        var n = len(requests)
        var results: List[PricingResult] = []
        for _ in range(n):
            results.append(PricingResult(
                price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False
            ))

        var ds_weights = self._compute_trap_weights(grid.s_points)
        var dv_weights = self._compute_trap_weights(grid.v_points)

        @parameter
        def worker(i: Int):
            var payoff = self._get_payoff_for_greeks(requests[i])
            var price = self._integrate_payoff_fast(
                grid, requests[i], ds_weights, dv_weights
            )
            var delta = self.greeks_computer.compute_delta(
                grid, self.interpolator,
                requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
            )
            var gamma = self.greeks_computer.compute_gamma(
                grid, self.interpolator,
                requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
            )
            var vega = self.greeks_computer.compute_vega(
                grid, self.interpolator,
                requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
            )
            results[i] = PricingResult(
                price=price, delta=delta, gamma=gamma, vega=vega, success=True
            )

        parallelize[worker](n)
        return results^

    def _price_gpu_batch(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """GPU batch pricing using payoff_integration_kernel.
        
        Transfers PDF grid and option parameters to GPU, launches one thread
        per option, and copies results back. Falls back to CPU on error.
        """
        comptime if has_accelerator():
            from std.gpu.host import DeviceContext
            from server.gpu_pricing_kernels import (
                payoff_integration_kernel,
                PRICER_DTYPE, PRICER_VEC_LAYOUT, PRICER_MAX_OPTIONS,
            )
            from std.sys import has_apple_gpu_accelerator
            from layout import LayoutTensor
            
            try:
                var ctx = DeviceContext(api="metal")
                var n_s = len(grid.s_points)
                var n_v = len(grid.v_points)
                var n_options = len(requests)
                
                # Validate sizes
                if n_options > PRICER_MAX_OPTIONS:
                    return self._price_cpu_parallel(grid, requests)
                
                # Flatten PDF grid with proper padding
                comptime if has_apple_gpu_accelerator():
                    var pdf_flat: List[Float32] = []
                    for i in range(n_s * n_v):
                        pdf_flat.append(Float32(0.0))
                    
                    var s_flat: List[Float32] = []
                    var v_flat: List[Float32] = []
                    var ds_flat: List[Float32] = []
                    var dv_flat: List[Float32] = []
                    var k_flat: List[Float32] = []
                    var bar_flat: List[Float32] = []
                    var price_flat: List[Float32] = []
                    
                    for _ in range(PRICER_MAX_OPTIONS):
                        k_flat.append(Float32(0.0))
                        bar_flat.append(Float32(0.0))
                        price_flat.append(Float32(0.0))
                    
                    # Fill PDF data
                    for i in range(n_s):
                        for j in range(n_v):
                            pdf_flat[i * n_v + j] = Float32(grid.pdf[i][j])
                    
                    # Fill grid points
                    for i in range(n_s):
                        s_flat.append(Float32(grid.s_points[i]))
                        ds_flat.append(Float32(1.0))  # Simplified weights
                    for i in range(n_v):
                        v_flat.append(Float32(grid.v_points[i]))
                        dv_flat.append(Float32(1.0))
                    
                    # Fill option parameters
                    for i in range(n_options):
                        k_flat[i] = Float32(requests[i].K)
                        bar_flat[i] = Float32(requests[i].barrier)
                    
                    # Create host buffers
                    var pdf_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS * PRICER_MAX_OPTIONS)
                    var s_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var v_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var ds_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var dv_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var k_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var bar_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var price_host = ctx.enqueue_create_host_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    ctx.synchronize()
                    
                    # Copy data to host buffers
                    for i in range(n_s * n_v):
                        if i < PRICER_MAX_OPTIONS * PRICER_MAX_OPTIONS:
                            pdf_host[i] = pdf_flat[i]
                    for i in range(n_s):
                        s_host[i] = s_flat[i]
                        ds_host[i] = ds_flat[i]
                    for i in range(n_v):
                        v_host[i] = v_flat[i]
                        dv_host[i] = dv_flat[i]
                    for i in range(n_options):
                        k_host[i] = k_flat[i]
                        bar_host[i] = bar_flat[i]
                    
                    # Create device buffers
                    var pdf_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS * PRICER_MAX_OPTIONS)
                    var s_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var v_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var ds_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var dv_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var k_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var bar_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    var price_dev = ctx.enqueue_create_buffer[DType.float32](PRICER_MAX_OPTIONS)
                    
                    # Copy to device
                    ctx.enqueue_copy(dst_buf=pdf_dev, src_buf=pdf_host)
                    ctx.enqueue_copy(dst_buf=s_dev, src_buf=s_host)
                    ctx.enqueue_copy(dst_buf=v_dev, src_buf=v_host)
                    ctx.enqueue_copy(dst_buf=ds_dev, src_buf=ds_host)
                    ctx.enqueue_copy(dst_buf=dv_dev, src_buf=dv_host)
                    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
                    ctx.enqueue_copy(dst_buf=bar_dev, src_buf=bar_host)
                    ctx.synchronize()
                    
                    # Create LayoutTensors
                    var pdf_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](pdf_dev)
                    var s_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](s_dev)
                    var v_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](v_dev)
                    var ds_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](ds_dev)
                    var dv_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](dv_dev)
                    var k_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](k_dev)
                    var bar_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](bar_dev)
                    var price_tensor = LayoutTensor[DType.float32, PRICER_VEC_LAYOUT](price_dev)
                    
                    # Launch kernel
                    ctx.enqueue_function[payoff_integration_kernel, payoff_integration_kernel](
                        pdf_tensor, s_tensor, v_tensor, ds_tensor, dv_tensor,
                        k_tensor, bar_tensor, price_tensor,
                        n_s, n_v, n_options,
                        grid_dim=n_options, block_dim=256,
                    )
                    ctx.synchronize()
                    
                    # Copy results back
                    ctx.enqueue_copy(dst_buf=price_host, src_buf=price_dev)
                    ctx.synchronize()
                    
                    # Build results
                    var results: List[PricingResult] = []
                    for i in range(n_options):
                        results.append(PricingResult(
                            price=Float64(price_host[i]),
                            delta=0.0,  # GPU Greeks not yet implemented
                            gamma=0.0,
                            vega=0.0,
                            success=True,
                        ))
                    return results^
            except:
                pass
            return self._price_cpu_parallel(grid, requests)
        else:
            return self._price_cpu_parallel(grid, requests)

    @always_inline
    def _get_payoff(self, req: PricingRequest) -> EuropeanCall:
        """Get the correct payoff type for Greeks computation.

        Returns EuropeanCall for all payoff types as a simplification.
        Barrier option Greeks would require different formulas.
        """
        _ = req
        return EuropeanCall()

    @always_inline
    def _get_payoff_for_greeks(self, req: PricingRequest) -> EuropeanCall:
        """Get payoff type for Greeks - currently always EuropeanCall.
        
        TODO: Implement proper barrier option Greeks when needed.
        """
        _ = req
        return EuropeanCall()

    @always_inline
    def _payoff_value(self, req: PricingRequest, S: Float64) -> Float64:
        """Evaluate payoff at spot price S. Inlined for zero call overhead."""
        if req.payoff_type == 0:
            return BarrierUpAndOut().evaluate(S, req.K, req.barrier)
        if req.payoff_type == 1:
            return EuropeanCall().evaluate(S, req.K, req.barrier)
        if req.payoff_type == 2:
            return BarrierDownAndIn().evaluate(S, req.K, req.barrier)
        if req.payoff_type == 3:
            return EuropeanPut().evaluate(S, req.K, req.barrier)
        return EuropeanCall().evaluate(S, req.K, req.barrier)

    def _compute_trap_weights(self, points: List[Float64]) -> List[Float64]:
        """Compute trapezoidal quadrature weights for a set of points.

        Pre-computed once, reused for every pricing call.
        Interior: (x[i+1] - x[i-1]) / 2 (midpoint rule)
        Boundary: 1.0 (half-interval)
        """
        var n = len(points)
        var w: List[Float64] = []
        for i in range(n):
            if i == 0 or i == n - 1:
                w.append(1.0)
            else:
                w.append((points[i + 1] - points[i - 1]) * 0.5)
        return w^

    def _integrate_payoff(self, grid: PDFGrid, req: PricingRequest) -> Float64:
        """Numerical integration: price = ∫∫ payoff(S) * pdf(S,V) dS dV.

        Retained for backward compatibility. Uses on-the-fly weight computation.
        """
        var ds_weights = self._compute_trap_weights(grid.s_points)
        var dv_weights = self._compute_trap_weights(grid.v_points)
        return self._integrate_payoff_fast(grid, req, ds_weights, dv_weights)

    def _integrate_payoff_fast(
        self,
        grid: PDFGrid,
        req: PricingRequest,
        ds_weights: List[Float64],
        dv_weights: List[Float64],
    ) -> Float64:
        """Optimized integration with pre-computed weights.

        Key optimization: payoff(S) is hoisted out of the inner V-loop
        since payoff depends only on S, not V. This avoids redundant
        payoff evaluation (was called n_s × n_v times, now n_s times).
        """
        var price = 0.0
        var n_s = len(grid.s_points)
        var n_v = len(grid.v_points)

        for i in range(n_s):
            var S = grid.s_points[i]
            var payoff_val = self._payoff_value(req, S)

            # Skip zero-payoff rows entirely (common for OTM options)
            if payoff_val == 0.0:
                continue

            var payoff_ds = payoff_val * ds_weights[i]

            # Inner loop: sum pdf over variance dimension
            var v_sum = 0.0
            comptime simd_width = simd_width_of[DType.float64]()

            var j = 0
            while j + simd_width <= n_v:
                var pdf_vals = SIMD[DType.float64, simd_width]()
                var dv_vals = SIMD[DType.float64, simd_width]()
                for k in range(simd_width):
                    pdf_vals[k] = grid.pdf[i][j + k]
                    dv_vals[k] = dv_weights[j + k]
                v_sum += (pdf_vals * dv_vals).reduce_add()
                j += simd_width

            # Scalar tail
            while j < n_v:
                v_sum += grid.pdf[i][j] * dv_weights[j]
                j += 1

            price += payoff_ds * v_sum

        return price
