# import_validator.py
# Comprehensive validation and fallback handling for LangChain/LangGraph imports

import logging
logger = logging.getLogger(__name__)
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class ImportStatus:
    """Track the status of package imports."""
    name: str
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None
    fallback_provided: bool = False

class LangChainImportValidator:
    """Validates and provides fallbacks for LangChain/LangGraph imports."""
    
    def __init__(self):
        self.import_status: Dict[str, ImportStatus] = {}
        self.logger = logger
        
    def validate_core_imports(self) -> Dict[str, ImportStatus]:
        """Validate all core LangChain/LangGraph imports."""
        
        # Define critical imports to check
        imports_to_check = [
            # LangGraph (critical)
            ("langgraph.graph", ["StateGraph", "START", "END"]),
            
            # LangChain Core (critical)
            ("langchain_core.messages", ["SystemMessage", "HumanMessage", "AIMessage"]),
            ("langchain_core.documents", ["Document"]),
            ("langchain_core.runnables", ["RunnableConfig"]),
            
            # LangChain Community (optional but important)
            ("langchain_community.utilities", ["GoogleSerperAPIWrapper"]),
            ("langchain_community.document_loaders", ["PyMuPDFLoader", "PyPDFLoader"]),
            ("langchain_community.vectorstores", ["FAISS"]),
            
            # LangChain Text Splitters
            ("langchain_text_splitters", ["RecursiveCharacterTextSplitter"]),
            
            # Provider-specific packages (Google GenAI only)
            ("langchain_google_genai", ["ChatGoogleGenerativeAI", "GoogleGenerativeAIEmbeddings"]),
        ]
        
        for module_name, imports in imports_to_check:
            status = self._check_import(module_name, imports)
            self.import_status[module_name] = status
            
        return self.import_status
    
    def _check_import(self, module_name: str, imports: List[str]) -> ImportStatus:
        """Check if a specific module and its imports are available."""
        try:
            module = __import__(module_name, fromlist=imports)
            
            # Check if specific imports are available
            missing_imports = []
            for import_name in imports:
                if not hasattr(module, import_name):
                    missing_imports.append(import_name)
            
            if missing_imports:
                error_msg = f"Missing imports: {missing_imports}"
                self.logger.warning(f"{module_name}: {error_msg}")
                return ImportStatus(module_name, False, error=error_msg)
            
            # Try to get version if available
            version = getattr(module, "__version__", None)
            
            self.logger.info(f"✅ {module_name} available" + (f" (v{version})" if version else ""))
            return ImportStatus(module_name, True, version=version)
            
        except ImportError as e:
            error_msg = str(e)
            self.logger.warning(f"❌ {module_name}: {error_msg}")
            return ImportStatus(module_name, False, error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"❌ {module_name}: {error_msg}")
            return ImportStatus(module_name, False, error=error_msg)
    
    def get_missing_packages(self) -> List[str]:
        """Get list of missing packages that should be installed."""
        missing = []
        
        for module_name, status in self.import_status.items():
            if not status.available:
                # Map module names to pip package names
                package_name = self._get_pip_package_name(module_name)
                if package_name:
                    missing.append(package_name)
                    
        return list(set(missing))  # Remove duplicates
    
    def _get_pip_package_name(self, module_name: str) -> Optional[str]:
        """Map module name to pip package name."""
        mapping = {
            "langgraph.graph": "langgraph",
            "langchain_core.messages": "langchain-core",
            "langchain_core.documents": "langchain-core", 
            "langchain_core.runnables": "langchain-core",
            "langchain_community.utilities": "langchain-community",
            "langchain_community.document_loaders": "langchain-community",
            "langchain_community.vectorstores": "langchain-community",
            "langchain_text_splitters": "langchain-text-splitters",
            "langchain_google_genai": "langchain-google-genai",
        }
        return mapping.get(module_name)
    
    def print_status_report(self):
        """Print a comprehensive status report."""
        print("\n" + "="*60)
        print("🔍 LANGCHAIN/LANGGRAPH IMPORT STATUS REPORT")
        print("="*60)
        
        # Categorize imports
        critical = []
        optional = []
        
        for module_name, status in self.import_status.items():
            if module_name in ["langgraph.graph", "langchain_core.messages", "langchain_core.documents"]:
                critical.append((module_name, status))
            else:
                optional.append((module_name, status))
        
        # Print critical imports
        print("\n🚨 CRITICAL IMPORTS:")
        for module_name, status in critical:
            icon = "✅" if status.available else "❌"
            version = f" (v{status.version})" if status.version else ""
            print(f"  {icon} {module_name}{version}")
            if not status.available and status.error:
                print(f"     Error: {status.error}")
        
        # Print optional imports
        print("\n🔧 OPTIONAL IMPORTS:")
        for module_name, status in optional:
            icon = "✅" if status.available else "⚠️"
            version = f" (v{status.version})" if status.version else ""
            print(f"  {icon} {module_name}{version}")
            if not status.available and status.error:
                print(f"     Error: {status.error}")
        
        # Print missing packages
        missing = self.get_missing_packages()
        if missing:
            print(f"\n📦 MISSING PACKAGES ({len(missing)}):")
            for package in missing:
                print(f"  - {package}")
            print(f"\n💡 Install with: pip install {' '.join(missing)}")
        else:
            print("\n✅ All packages are available!")
        
        print("="*60)
    
    def get_installation_command(self) -> Optional[str]:
        """Get pip installation command for missing packages."""
        missing = self.get_missing_packages()
        if missing:
            return f"pip install {' '.join(missing)}"
        return None


def validate_imports() -> LangChainImportValidator:
    """Main function to validate all imports."""
    validator = LangChainImportValidator()
    validator.validate_core_imports()
    return validator


def create_safe_fallbacks():
    """Create safe fallback classes for missing imports."""
    
    # Fallback for LangGraph
    if "langgraph.graph" not in sys.modules:
        class StateGraph:
            def __init__(self, state_type=None):
                self.nodes = {}
                logging.warning("Using fallback StateGraph - LangGraph not available")
            def add_node(self, name, func): pass
            def add_edge(self, a, b): pass
            def add_conditional_edges(self, node, route_fn, mapping): pass
            def compile(self): return None
        
        # Add to global namespace
        globals()['StateGraph'] = StateGraph
        globals()['START'] = "__START__"
        globals()['END'] = "__END__"
    
    # Fallback for LangChain messages
    if "langchain_core.messages" not in sys.modules:
        class SystemMessage:
            def __init__(self, content): self.content = content
        class HumanMessage:
            def __init__(self, content): self.content = content
        class AIMessage:
            def __init__(self, content): self.content = content
        
        globals()['SystemMessage'] = SystemMessage
        globals()['HumanMessage'] = HumanMessage
        globals()['AIMessage'] = AIMessage
    
    # Fallback for Documents
    if "langchain_core.documents" not in sys.modules:
        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}
        
        globals()['Document'] = Document


if __name__ == "__main__":
    # Run validation when script is executed directly
    validator = validate_imports()
    validator.print_status_report()
    
    cmd = validator.get_installation_command()
    if cmd:
        print(f"\n🔧 To install missing packages, run:")
        print(f"   {cmd}")