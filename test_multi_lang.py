import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT))

from extractors.java_csharp_extractor import JavaCSharpExtractor
from extractors.js_ts_extractor import JSTSExtractor
from extractors.cpp_extractor import CppExtractor
from extractors.html_css_extractor import HtmlCssExtractor
from models import NodeType

WORK_DIR = PROJECT / ".test_multi_lang"

# ── Test Cases ──

JAVA_SRC = """
package com.example;
public class UserService {
    public void process(String data) {}
    public void process(int id) {}
}
"""

CSHARP_SRC = """
namespace Example {
    public interface IService {
        void Execute();
    }
}
"""

JS_SRC = """
const myCallback = () => { console.log('hello'); };
function main() {
    setTimeout(() => {
        myCallback();
    }, 1000);
}
"""

TS_SRC = """
interface User { id: number; }
type Status = "active" | "inactive";
"""

CPP_SRC = """
#define MAX(a,b) ((a) > (b) ? (a) : (b))
namespace Core {
    namespace Math {
        class Calculator {
            int add(int a, int b) { return a + b; }
        };
    }
}
"""

HTML_SRC = """
<div id="main-container" class="layout grid">
    <button id="submitBtn" class="btn primary">Submit</button>
</div>
"""

CSS_SRC = """
#main-container { display: flex; }
.btn.primary { color: white; }
"""

def write_file(rel_path: str, content: str) -> Path:
    p = WORK_DIR / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip(), encoding="utf-8")
    return p

def check(name: str, condition: bool, msg: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        if msg:
            print(f"        {msg}")
        global failed
        failed += 1

passed = 0
failed = 0

def main():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()

    print("=== Java & C# ===")
    java_file = write_file("UserService.java", JAVA_SRC)
    cs_file = write_file("IService.cs", CSHARP_SRC)
    
    java_ext = JavaCSharpExtractor(WORK_DIR)
    j_nodes, j_edges = java_ext.parse_file(java_file, JAVA_SRC)
    
    check("Java: Extracted overloaded methods", any("process(String)" in n.id for n in j_nodes) and any("process(int)" in n.id for n in j_nodes), str([n.id for n in j_nodes]))
    
    cs_nodes, cs_edges = java_ext.parse_file(cs_file, CSHARP_SRC)
    check("C#: Extracted interface", any(n.metadata.get("is_interface") for n in cs_nodes), str([n.metadata for n in cs_nodes]))

    print("\n=== JS & TS ===")
    js_file = write_file("app.js", JS_SRC)
    ts_file = write_file("types.ts", TS_SRC)
    
    js_ext = JSTSExtractor(WORK_DIR)
    js_nodes, js_edges = js_ext.parse_file(js_file, JS_SRC)
    
    check("JS: Extracted assigned arrow function", any("myCallback" in n.id for n in js_nodes), str([n.id for n in js_nodes]))
    check("JS: Extracted anonymous callback", any("anon_func" in n.id for n in js_nodes), str([n.id for n in js_nodes]))
    
    ts_nodes, ts_edges = js_ext.parse_file(ts_file, TS_SRC)
    check("TS: Extracted type and interface", any(n.metadata.get("is_type_definition") for n in ts_nodes), str([n.metadata for n in ts_nodes]))

    print("\n=== C & C++ ===")
    cpp_file = write_file("math.cpp", CPP_SRC)
    cpp_ext = CppExtractor(WORK_DIR)
    cpp_nodes, cpp_edges = cpp_ext.parse_file(cpp_file, CPP_SRC)
    
    check("C++: Extracted macro as function", any(n.metadata.get("is_macro") for n in cpp_nodes), str([n.metadata for n in cpp_nodes]))
    check("C++: Correct namespace in ID", any("Core::Math::Calculator" in n.id for n in cpp_nodes), str([n.id for n in cpp_nodes]))

    print("\n=== HTML & CSS ===")
    html_file = write_file("index.html", HTML_SRC)
    css_file = write_file("style.css", CSS_SRC)
    
    html_ext = HtmlCssExtractor(WORK_DIR)
    h_nodes, h_edges = html_ext.parse_file(html_file, HTML_SRC)
    c_nodes, c_edges = html_ext.parse_file(css_file, CSS_SRC)
    
    check("HTML: Extracted #submitBtn", any("#submitBtn" in n.id for n in h_nodes), str([n.id for n in h_nodes]))
    check("HTML: Extracted .btn class", any(".btn" in n.id for n in h_nodes), str([n.id for n in h_nodes]))
    check("CSS: Extracted #main-container selector", any("#main-container" in n.id for n in c_nodes), str([n.id for n in c_nodes]))

    print(f"\nResults: {failed} failed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
