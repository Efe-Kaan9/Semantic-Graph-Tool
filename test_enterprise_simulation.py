import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT))

from core.database import CodeGraphDB
from core.jit_sync import JITSyncManager

WORK_DIR = PROJECT / ".enterprise_sim"

# ── 1. Legacy Spaghetti C Code (Monolithic, no SOLID, Macros) ──
LEGACY_C_SRC = """
#include <stdio.h>
#include <string.h>

#define MAX_USERS 1000
#define DO_AUTH(u, p) ((strcmp(u,"admin")==0 && strcmp(p,"1234")==0) ? 1 : 0)

int global_state = 0;

void process_everything_at_once(char* user, char* pass, int action) {
    if (action == 1) {
        if (DO_AUTH(user, pass)) {
            global_state = 1;
            // 500 lines of spaghetti omitted
            goto success;
        } else {
            goto fail;
        }
    } else if (action == 2) {
        // DB connection parsing inside a random function
        char* db_conn_str = "Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;";
        printf("Connecting to %s\\n", db_conn_str);
    }
    
success:
    printf("Success\\n");
    return;
fail:
    printf("Fail\\n");
    return;
}
"""

# ── 2. Modern Java Code (SOLID, LLM-written, well commented) ──
MODERN_JAVA_SRC = """
package com.enterprise.auth;

/**
 * Service responsible for orchestrating the user authentication flow.
 * Adheres to the Single Responsibility Principle (SRP) by delegating
 * token generation and database validation to specialized providers.
 */
public class ModernUserService {
    
    private final TokenProvider tokenProvider;
    private final UserRepository repository;
    
    public ModernUserService(TokenProvider tokenProvider, UserRepository repository) {
        this.tokenProvider = tokenProvider;
        this.repository = repository;
    }
    
    /**
     * Authenticates a user and returns a JWT token.
     * @param username The user's identifier
     * @param password The user's secret
     * @return A valid JWT token if successful, null otherwise
     */
    public String authenticateUser(String username, String password) {
        User user = repository.findByUsername(username);
        if (user != null && user.verifyPassword(password)) {
            return tokenProvider.generateToken(user);
        }
        return null;
    }
}
"""

# ── 3. TypeScript vs JS - Semantic Collision ──
# Both handle auth, testing if vector search returns close scores but distinguishes them.
TS_AUTH_SRC = """
export class AuthManager {
    public validateCredentials(user: string, pass: string): boolean {
        // High security enterprise validation
        return user.length > 0 && pass.length > 8;
    }
}
"""

JS_AUTH_SRC = """
// Legacy JS helper for authentication validation
function checkAuthCredentials(user, pass) {
    // Basic validation
    return user !== "" && pass !== "";
}
"""

# ── 4. Python God Class (Anti-pattern) ──
GOD_CLASS_PY_SRC = """
class EnterpriseSystemCore:
    def __init__(self):
        self.db = None
        self.ui = None
        self.network = None
        
    def do_login(self, u, p):
        pass
        
    def render_ui(self):
        pass
        
    def calculate_taxes(self):
        pass
        
    def parse_database_connection_string(self, raw_config):
        # Buried database logic in a God class
        parts = raw_config.split(";")
        return {p.split("=")[0]: p.split("=")[1] for p in parts if "=" in p}
"""

def write_file(rel_path: str, content: str) -> Path:
    p = WORK_DIR / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip(), encoding="utf-8")
    return p

def main():
    print("Initializing Enterprise Environment Simulation...")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()

    write_file("legacy/spaghetti_core.c", LEGACY_C_SRC)
    write_file("src/main/java/com/enterprise/auth/ModernUserService.java", MODERN_JAVA_SRC)
    write_file("frontend/auth_manager.ts", TS_AUTH_SRC)
    write_file("scripts/auth_helpers.js", JS_AUTH_SRC)
    write_file("backend/god_class.py", GOD_CLASS_PY_SRC)

    print("Files generated. Spinning up CodeGraphDB and JITSyncManager...")
    
    # Initialize DB (creates SQLite and ChromaDB inside WORK_DIR/.code_graph)
    db = CodeGraphDB(WORK_DIR)
    manager = JITSyncManager(WORK_DIR, db, WORK_DIR / ".code_graph")
    
    # Run cold start sync
    sync_res = manager.sync()
    print(f"Sync Complete in {sync_res.elapsed_ms:.2f}ms. Added {len(sync_res.added)} files.")
    print(f"Total Nodes in DB: {db.node_count}")

    print("\\nTEST 1: Semantic Collision (Auth Process)")
    print("Query: 'user login authentication process verification'")
    res1 = db.semantic_search("user login authentication process verification", n_results=4, node_type="METHOD")
    for r in res1:
        print(f"  [{r['score']:.4f}] {r['id']} (Type: {r['type']})")

    print("\\nTEST 2: Finding Buried Logic (DB Connection)")
    print("Query: 'parse database connection string'")
    res2 = db.semantic_search("parse database connection string", n_results=3)
    for r in res2:
        print(f"  [{r['score']:.4f}] {r['id']} (Type: {r['type']})")
        
    print("\\nTEST 3: Extracting Legacy Macro (C)")
    res3 = db.semantic_search("DO_AUTH", n_results=1)
    if res3:
        # Since semantic search returns a flat dict, we need to fetch the raw node for metadata
        node_raw = db.get_node(res3[0]['id'])
        is_macro = node_raw.get("metadata", {}).get("is_macro") if node_raw else None
        print(f"  Found Macro: {res3[0]['id']} (Is Macro: {is_macro})")
        
    print("\\nSimulation Completed Successfully!")

if __name__ == "__main__":
    main()
