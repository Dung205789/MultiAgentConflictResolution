from typing import Dict, List, Any, Optional, Tuple

from src.memory.shared_memory_store import SharedMemoryStore
from src.utils.retriever import retrieve_topk
from src.utils.extractor import extract_memories_from_text


class AgentRuntime:
    """
    Agent-level runtime for multi-agent shared memory operations.
    Provides read/retrieve/propose abstractions with model-backed capabilities.
    """

    def __init__(
        self, 
        store: SharedMemoryStore, 
        agent_id: str,
        mode: str = "debug_fallback"
    ):
        """
        Initialize agent runtime.
        
        Args:
            store: Shared memory store
            agent_id: Unique agent identifier
            mode: Either "research_strict" (fail if models required but unavailable) or 
                  "debug_fallback" (allow fallback to rule-based)
        """
        self.store = store
        self.agent_id = agent_id
        self.mode = mode
        self.read_snapshot_time = 0.0
        
        # Validate mode
        if mode not in ["research_strict", "debug_fallback"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'research_strict' or 'debug_fallback'")

    def read(self, query: str = None, method: str = "hybrid", k: int = 5) -> List[Dict[str, Any]]:
        """
        Read from shared memory, optionally filtered by query.
        
        Args:
            query: Optional search query
            method: Retrieval method ("keyword", "embedding", "hybrid")
            k: Number of results to return
            
        Returns:
            List of memory entries
        """
        import time
        self.read_snapshot_time = time.time()
        
        if query:
            results = retrieve_topk(
                [r.to_dict() for r in self.store.get_all_visible()],
                query=query,
                k=k,
                method=method,
                mode=self.mode
            )
            return [r[0] for r in results]
        else:
            return [r.to_dict() for r in self.store.get_all_visible()]

    def extract_from_text(self, text: str, use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Extract structured memories from text.
        
        Args:
            text: Input text
            use_llm: Whether to use LLM-based extraction
            
        Returns:
            List of extracted memory items
        """
        return extract_memories_from_text(text, use_llm=use_llm, mode=self.mode)

    def propose_write(self, proposal: Dict[str, Any], writer_type: str = "conflict_aware") -> Dict[str, Any]:
        """
        Propose a write to shared memory.
        
        Args:
            proposal: Memory entry to write
            writer_type: Writer type ("conflict_aware", "lww", "naive")
            
        Returns:
            Write result
        """
        from src.conflict.conflict_aware_writer import ConflictAwareWriter
        from src.conflict.baselines import LastWriteWinsWriter, NaiveAppendWriter
        from src.conflict.staleness_detector import StalenessDetector
        
        if writer_type == "conflict_aware":
            writer = ConflictAwareWriter(self.store, StalenessDetector())
            return writer.write(proposal, agent_id=self.agent_id, read_snapshot_time=self.read_snapshot_time)
        elif writer_type == "lww":
            writer = LastWriteWinsWriter(self.store)
            return writer.write(proposal, agent_id=self.agent_id)
        elif writer_type == "naive":
            writer = NaiveAppendWriter(self.store)
            return writer.write(proposal, agent_id=self.agent_id)
        else:
            raise ValueError(f"Unknown writer_type: {writer_type}")
