import os
import re

utils_list = {
    "_zeros": ("zeros", "numerics.utils"),
    "_zeros_mat": ("zeros_mat", "numerics.utils"),
    "_zeros_2d": ("zeros_mat", "numerics.utils"),
    "_zeros_3d": ("zeros_3d", "numerics.utils"), 
    "_abs": ("abs_f64", "numerics.utils"),
    "_max": ("max_f64", "numerics.utils"),
    "_min": ("min_f64", "numerics.utils"),
    "_copy_vec": ("copy_vec", "numerics.utils"),
    "_pow_pos": ("pow_pos", "numerics.utils"),
    "_linspace": ("linspace", "numerics.utils"),
    "_matmul_vec": (None, None), # delete entirely
}

def clean_file(path):
    with open(path, 'r') as f:
        content = f.read()

    new_content = content
    imports_to_add = set()

    for fn_name, (util_name, module_path) in utils_list.items():
        # find function definition
        pattern = re.compile(rf"(@always_inline\n)?def {fn_name}\(.*?\) -> .*?:\n(?:    .*\n)*", re.MULTILINE)
        if pattern.search(new_content):
            new_content = pattern.sub("", new_content)
            if util_name:
                imports_to_add.add((module_path, util_name))
        
        # replace calls
        if util_name and fn_name != util_name:
            # specifically replace independent calls to fn_name
            # e.g., _zeros( -> zeros(
            new_content = re.sub(rf"\b{fn_name}\(", f"{util_name}(", new_content)

    if imports_to_add and new_content != content:
        # Group imports by module
        by_module = {}
        for mod, name in imports_to_add:
            by_module.setdefault(mod, set()).add(name)
        
        import_statements = "\n".join([f"from {mod} import {', '.join(sorted(names))}" for mod, names in by_module.items()])
        
        # Insert after the module docstring or at top
        if new_content.startswith('"""'):
            end_idx = new_content.find('"""', 3)
            if end_idx != -1:
                end_idx += 3
                new_content = new_content[:end_idx] + "\n\n" + import_statements + new_content[end_idx:]
            else:
                new_content = import_statements + "\n\n" + new_content
        else:
            new_content = import_statements + "\n\n" + new_content

    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Cleaned {path}")


src_dir = "src"
for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith(".mojo") and f != "utils.mojo":
            clean_file(os.path.join(root, f))

