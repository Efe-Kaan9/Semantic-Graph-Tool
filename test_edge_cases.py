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

WORK_DIR = PROJECT / ".test_edge_cases"

# ── 1. Java / C# Edge Cases ──
# - Inner classes
# - Generics in overloads
# - Array and varargs parameters
JAVA_EDGE_SRC = """
package com.complex;
public class OuterClass<T> {
    public interface IProcessor {
        void process(T[] items, String... args);
    }
    
    private class InnerClass implements IProcessor {
        @Override
        public void process(T[] items, String... args) {
            // ...
        }
        
        public void process(List<Map<String, T>> complexArg) {
            // ...
        }
    }
}
"""

# ── 2. JS / TS Edge Cases ──
# - IIFE (Immediately Invoked Function Expression)
# - Higher order functions (function returning function)
# - Complex Intersection Types
TS_EDGE_SRC = """
type ComplexAlias = { id: string } & ( { status: "active" } | { status: "inactive", reason: string } );

const createFactory = (config: any) => {
    return function innerWorker() {
        const iifeResult = (() => {
            console.log("IIFE executed");
            return 42;
        })();
    };
};
"""

# ── 3. C / C++ Edge Cases ──
# - Multi-line Macros
# - Nested and anonymous namespaces
# - Template classes
CPP_EDGE_SRC = """
#define COMPLEX_MACRO(x, y) \\
    do { \\
        if ((x) > (y)) \\
            printf("Max is %d\\n", (x)); \\
        else \\
            printf("Max is %d\\n", (y)); \\
    } while (0)

namespace Level1 {
    namespace { // Anonymous namespace
        void hidden_function() {}
    }
    namespace Level2 {
        template <typename T>
        class TemplateClass {
            T compute(T a, T* b) { return a + *b; }
        };
    }
}
"""

# ── 4. HTML / CSS Edge Cases ──
# - Multiple classes, messy spacing
# - Complex CSS selectors
HTML_CSS_EDGE_SRC = """
<!-- HTML -->
<div id = " messy-id " class=" class1   class2 " data-custom="value">
    <span id='single-quote-id'></span>
</div>
"""
CSS_EDGE_SRC = """
/* CSS */
div#messy-id > .class1.class2:hover::before, 
#single-quote-id[data-custom="value"] {
    display: none !important;
}
"""

def write_file(rel_path: str, content: str) -> Path:
    p = WORK_DIR / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip(), encoding="utf-8")
    return p

passed = 0
failed = 0

def check(name: str, condition: bool, msg: str = "") -> None:
    if condition:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        if msg:
            print(f"         Context: {msg}")
        global failed
        failed += 1

def main():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()

    print("=== Java Edge Cases ===")
    java_file = write_file("Complex.java", JAVA_EDGE_SRC)
    java_ext = JavaCSharpExtractor(WORK_DIR)
    j_nodes, j_edges = java_ext.parse_file(java_file, JAVA_EDGE_SRC)
    j_ids = [n.id for n in j_nodes]
    
    check("Inner class extracted", any("OuterClass" in i and "InnerClass" in i for i in j_ids), str(j_ids))
    check("Complex parameter overload 1 extracted", any("process" in i for i in j_ids), str(j_ids))
    
    print("\n=== TS Edge Cases ===")
    ts_file = write_file("complex.ts", TS_EDGE_SRC)
    ts_ext = JSTSExtractor(WORK_DIR)
    t_nodes, t_edges = ts_ext.parse_file(ts_file, TS_EDGE_SRC)
    t_ids = [n.id for n in t_nodes]
    
    check("Complex type alias extracted", any("ComplexAlias" in i for i in t_ids), str(t_ids))
    check("Higher order function extracted", any("createFactory" in i for i in t_ids), str(t_ids))
    check("Inner worker function extracted", any("innerWorker" in i for i in t_ids), str(t_ids))
    check("IIFE (anonymous) extracted", any("anon_func" in i for i in t_ids), str(t_ids))

    print("\n=== C++ Edge Cases ===")
    cpp_file = write_file("complex.cpp", CPP_EDGE_SRC)
    cpp_ext = CppExtractor(WORK_DIR)
    c_nodes, c_edges = cpp_ext.parse_file(cpp_file, CPP_EDGE_SRC)
    c_ids = [n.id for n in c_nodes]
    
    check("Multi-line macro extracted", any("COMPLEX_MACRO" in i for i in c_ids), str(c_ids))
    check("Anonymous namespace func extracted", any("hidden_function" in i for i in c_ids), str(c_ids))
    check("Nested template class extracted", any("Level1::Level2::TemplateClass" in i for i in c_ids), str(c_ids))
    check("Template method extracted", any("compute" in i for i in c_ids), str(c_ids))

    print("\n=== HTML/CSS Edge Cases ===")
    html_file = write_file("messy.html", HTML_CSS_EDGE_SRC)
    css_file = write_file("messy.css", CSS_EDGE_SRC)
    h_ext = HtmlCssExtractor(WORK_DIR)
    h_nodes, h_edges = h_ext.parse_file(html_file, HTML_CSS_EDGE_SRC)
    cs_nodes, cs_edges = h_ext.parse_file(css_file, CSS_EDGE_SRC)
    
    h_ids = [n.id for n in h_nodes]
    c_ids = [n.id for n in cs_nodes]
    
    check("HTML: Messy ID spacing stripped", any("#messy-id" in i for i in h_ids), str(h_ids))
    check("HTML: Multiple classes split correctly", any(".class1" in i for i in h_ids) and any(".class2" in i for i in h_ids), str(h_ids))
    check("HTML: Single quote ID extracted", any("#single-quote-id" in i for i in h_ids), str(h_ids))
    check("CSS: Complex selector extracted", any("div#messy-id > .class1.class2:hover::before,  #single-quote-id[data-custom=\"value\"]" in i for i in c_ids), str(c_ids))

    print(f"\nFinal Result: {failed} Failures.")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
