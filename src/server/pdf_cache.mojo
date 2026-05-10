from std.python import Python, PythonObject


@fieldwise_init
struct PDFGrid(Copyable, Movable):
    """Pre-computed PDF grid p(S,V,T) from FPE solver."""

    var pdf: List[List[Float64]]
    var s_points: List[Float64]
    var v_points: List[Float64]
    var T: Float64
    var ds_weights: List[Float64]
    var dv_weights: List[Float64]

    def precompute_weights(mut self):
        """Compute once, reuse for every pricing call."""
        # ds
        for i in range(len(self.s_points)):
            if i == 0 or i == len(self.s_points) - 1:
                self.ds_weights.append(1.0)
            else:
                self.ds_weights.append(
                    (self.s_points[i + 1] - self.s_points[i - 1]) * 0.5
                )
        # dv
        for i in range(len(self.v_points)):
            if i == 0 or i == len(self.v_points) - 1:
                self.dv_weights.append(1.0)
            else:
                self.dv_weights.append(
                    (self.v_points[i + 1] - self.v_points[i - 1]) * 0.5
                )

    def to_python_object(self) raises -> PythonObject:
        """Convert grid to Python dict for serialization."""
        var py_pdf = Python.list()
        for i in range(len(self.pdf)):
            var row = Python.list()
            for j in range(len(self.pdf[i])):
                _ = row.append(PythonObject(self.pdf[i][j]))
            _ = py_pdf.append(row)

        var py_s = Python.list()
        for i in range(len(self.s_points)):
            _ = py_s.append(PythonObject(self.s_points[i]))

        var py_v = Python.list()
        for i in range(len(self.v_points)):
            _ = py_v.append(PythonObject(self.v_points[i]))

        var py_ds = Python.list()
        for i in range(len(self.ds_weights)):
            _ = py_ds.append(PythonObject(self.ds_weights[i]))

        var py_dv = Python.list()
        for i in range(len(self.dv_weights)):
            _ = py_dv.append(PythonObject(self.dv_weights[i]))

        return Python.dict(
            pdf=py_pdf,
            s_points=py_s,
            v_points=py_v,
            T=PythonObject(self.T),
            ds_weights=py_ds,
            dv_weights=py_dv,
        )

    @staticmethod
    def from_python_object(py_obj: PythonObject) raises -> PDFGrid:
        """Reconstruct grid from Python dict."""
        var pdf: List[List[Float64]] = []
        var py_pdf = py_obj["pdf"]
        for i in range(len(py_pdf)):
            var row: List[Float64] = []
            for j in range(len(py_pdf[i])):
                row.append(Float64(py=py_pdf[i][j]))
            pdf.append(row^)

        var s_points: List[Float64] = []
        var py_s = py_obj["s_points"]
        for i in range(len(py_s)):
            s_points.append(Float64(py=py_s[i]))

        var v_points: List[Float64] = []
        var py_v = py_obj["v_points"]
        for i in range(len(py_v)):
            v_points.append(Float64(py=py_v[i]))

        var ds_weights: List[Float64] = []
        var py_ds = py_obj["ds_weights"]
        for i in range(len(py_ds)):
            ds_weights.append(Float64(py=py_ds[i]))

        var dv_weights: List[Float64] = []
        var py_dv = py_obj["dv_weights"]
        for i in range(len(py_dv)):
            dv_weights.append(Float64(py=py_dv[i]))

        return PDFGrid(
            pdf=pdf^,
            s_points=s_points^,
            v_points=v_points^,
            T=Float64(py=py_obj["T"]),
            ds_weights=ds_weights^,
            dv_weights=dv_weights^,
        )


struct PDFCache:
    """Cache of pre-computed PDF grids indexed by parameter hash."""

    var grids: Dict[UInt64, PDFGrid]

    def __init__(out self):
        self.grids = {}

    def store(mut self, param_hash: UInt64, var grid: PDFGrid):
        grid.precompute_weights()
        self.grids[param_hash] = grid^

    def get(self, param_hash: UInt64) -> Optional[PDFGrid]:
        return self.grids.get(param_hash)

    def contains(self, param_hash: UInt64) -> Bool:
        return param_hash in self.grids

    def save_to_disk(self, path: String) raises -> Bool:
        """Serialize cached PDF grids to disk using Python json (safe, no code execution).
        """
        var json_mod = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var data = Python.dict()
        for entry in self.grids.items():
            var key_int = Int(py=PythonObject(entry.key))
            data[key_int] = entry.value.to_python_object()

        var filepath = path + "/fpe_pdf_cache.json"
        try:
            var f = builtins.open(String(filepath), "w")
            f.write(String(py=json_mod.dumps(data)))
            f.close()
            return True
        except e:
            return False

    def load_from_disk(mut self, path: String) raises -> Bool:
        """Deserialize PDF grids from disk using Python json (safe, no code execution).
        """
        var json_mod = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var filepath = path + "/fpe_pdf_cache.json"
        try:
            var f = builtins.open(String(filepath), "r")
            var content = f.read()
            f.close()
            var data = json_mod.loads(String(py=content))
            for key_int in data:
                var key = UInt64(Int(py=key_int))
                var grid = PDFGrid.from_python_object(data[key_int])
                self.grids[key] = grid^
            return True
        except e:
            return False
