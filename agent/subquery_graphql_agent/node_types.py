"""
Node Type Detection and Schema Analysis Module

Handles detection and analysis of different GraphQL node types (SubQL vs The Graph)
and provides appropriate schema parsing logic for each.
"""

import re
from typing import Dict, Any, Optional, List, Tuple


class GraphqlProvider:
    """Supported GraphQL provider types."""
    SUBQL = "subql"
    THE_GRAPH = "thegraph"
    CODEX = "codex"
    UNKNOWN = "unknown"
    
    @classmethod
    def all_values(cls) -> List[str]:
        """Get all valid provider type values."""
        return [cls.SUBQL, cls.THE_GRAPH, cls.UNKNOWN]

class GraphqlProviderDetector:
    """Detects node type based on manifest and schema content."""
    
    @staticmethod
    def detect_from_manifest(manifest: Dict[str, Any]) -> str:
        """
        Detect node type from project manifest.
        
        Args:
            manifest: Project manifest dictionary
            
        Returns:
            str: Detected node type
        """
        # Check for SubQL-specific fields
        runner = manifest.get('runner')
        if runner:
            # SubQL projects have runner field with @subql/ packages
            node_name = runner.get('node', {}).get('name', '') if isinstance(runner.get('node'), dict) else ''
            query_name = runner.get('query', {}).get('name', '') if isinstance(runner.get('query'), dict) else ''
            
            if node_name.startswith('@subql/') or query_name.startswith('@subql/'):
                return GraphqlProvider.SUBQL
        
        # Check for file path formats to distinguish between SubQL and The Graph
        schema_info = manifest.get('schema', {})
        if isinstance(schema_info, dict) and 'file' in schema_info:
            file_info = schema_info['file']
            
            # The Graph format: { "/": "/ipfs/QmXXX" }
            if isinstance(file_info, dict) and '/' in file_info:
                file_path = file_info.get('/')
                if file_path and file_path.startswith('/ipfs/'):
                    return GraphqlProvider.THE_GRAPH
            
            # SubQL format: "ipfs://QmXXX" or simple file path
            elif isinstance(file_info, str):
                if file_info.startswith('ipfs://'):
                    return GraphqlProvider.SUBQL
        
        # If no runner field and uses The Graph file format, it's The Graph
        if not runner and schema_info:
            return GraphqlProvider.THE_GRAPH
        
        return GraphqlProvider.UNKNOWN


def detect_node_type(manifest: Dict[str, Any]) -> str:
    """
    Detect node type using manifest only.
    
    Args:
        manifest: Project manifest dictionary
        schema_content: Raw GraphQL schema string (used for entity analysis only)
        
    Returns:
        Tuple of (detected_node_type, analysis_metadata)
    """
    # Use manifest as the definitive source for node type detection
    detected_type = GraphqlProviderDetector.detect_from_manifest(manifest)
    
    return detected_type