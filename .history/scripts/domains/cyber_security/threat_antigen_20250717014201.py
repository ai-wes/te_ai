"""
Threat Antigen Module for Cybersecurity
======================================

Converts cybersecurity threats into graph representations that can be
processed by the TE-AI system, similar to biological antigens.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

from scripts.core.antigen import Antigen


@dataclass
class ThreatIndicator:
    """Represents an indicator of compromise (IoC)"""
    indicator_type: str  # 'ip', 'domain', 'hash', 'pattern'
    value: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    associated_malware: Optional[List[str]] = None
    
    def to_embedding(self) -> torch.Tensor:
        """Convert indicator to embedding"""
        # Simple hash-based embedding
        hash_obj = hashlib.sha256(f"{self.indicator_type}:{self.value}".encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to normalized float values
        embedding = torch.tensor([b / 255.0 for b in hash_bytes[:32]], dtype=torch.float32)
        
        # Add metadata features
        type_encoding = {
            'ip': 0.25, 'domain': 0.5, 'hash': 0.75, 'pattern': 1.0
        }
        metadata = torch.tensor([
            type_encoding.get(self.indicator_type, 0.0),
            self.confidence,
            (datetime.now() - self.first_seen).days / 365.0,  # Age in years
            (datetime.now() - self.last_seen).days / 30.0     # Recency in months
        ], dtype=torch.float32)
        
        return torch.cat([embedding, metadata])


class ThreatAntigen(Antigen):
    """Represents a cybersecurity threat as an antigen for the immune system"""
    
    def __init__(self, 
                 threat_name: str,
                 threat_type: str,
                 indicators: List[ThreatIndicator],
                 attack_vector: Optional[str] = None,
                 severity: float = 0.5):
        super().__init__(name=threat_name)
        self.threat_type = threat_type
        self.indicators = indicators
        self.attack_vector = attack_vector
        self.severity = severity
        
        # Generate feature matrix from indicators
        self.feature_matrix = self._generate_features()
        
    def _generate_features(self) -> torch.Tensor:
        """Generate feature matrix from threat indicators"""
        if not self.indicators:
            # Default feature if no indicators
            return torch.randn(10, 36)  # 10 nodes, 36 features
        
        # Convert each indicator to features
        features = []
        for indicator in self.indicators:
            features.append(indicator.to_embedding())
        
        # Stack and pad if necessary
        feature_matrix = torch.stack(features)
        
        # Ensure consistent dimensionality
        if feature_matrix.shape[1] < 36:
            padding = torch.zeros(feature_matrix.shape[0], 36 - feature_matrix.shape[1])
            feature_matrix = torch.cat([feature_matrix, padding], dim=1)
        else:
            feature_matrix = feature_matrix[:, :36]
        
        return feature_matrix
    
    def to_graph(self) -> Data:
        """Convert threat to graph representation"""
        num_nodes = len(self.indicators) if self.indicators else 10
        
        # Node features
        x = self.feature_matrix
        
        # Create edges based on indicator relationships
        edge_index = []
        
        # Connect indicators that share characteristics
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if self._indicators_related(i, j):
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Bidirectional
        
        # If no edges, create a minimal connected graph
        if not edge_index:
            for i in range(num_nodes - 1):
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Edge attributes based on relationship strength
        edge_attr = torch.ones(edge_index.shape[1], 1)
        
        # Global features
        global_features = torch.tensor([
            self.severity,
            self._encode_threat_type(),
            self._encode_attack_vector(),
            float(len(self.indicators))
        ], dtype=torch.float32)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            y=torch.tensor([self.severity], dtype=torch.float32)
        )
    
    def _indicators_related(self, i: int, j: int) -> bool:
        """Check if two indicators are related"""
        if i >= len(self.indicators) or j >= len(self.indicators):
            return False
        
        ind1, ind2 = self.indicators[i], self.indicators[j]
        
        # Same type of indicator
        if ind1.indicator_type == ind2.indicator_type:
            return True
        
        # Temporal proximity
        time_diff = abs((ind1.first_seen - ind2.first_seen).days)
        if time_diff < 7:  # Within a week
            return True
        
        # Shared malware associations
        if ind1.associated_malware and ind2.associated_malware:
            if set(ind1.associated_malware) & set(ind2.associated_malware):
                return True
        
        return False
    
    def _encode_threat_type(self) -> float:
        """Encode threat type as numeric value"""
        threat_types = {
            'malware': 0.1, 'ransomware': 0.2, 'trojan': 0.3,
            'ddos': 0.4, 'phishing': 0.5, 'apt': 0.6,
            'zero_day': 0.7, 'botnet': 0.8, 'unknown': 0.9
        }
        return threat_types.get(self.threat_type.lower(), 0.9)
    
    def _encode_attack_vector(self) -> float:
        """Encode attack vector as numeric value"""
        if not self.attack_vector:
            return 0.0
        
        vectors = {
            'network': 0.2, 'email': 0.4, 'web': 0.6,
            'usb': 0.8, 'supply_chain': 1.0
        }
        return vectors.get(self.attack_vector.lower(), 0.0)
    
    def get_complexity_score(self) -> float:
        """Calculate threat complexity score"""
        base_score = self.severity
        
        # Factor in number of indicators
        indicator_factor = min(len(self.indicators) / 100.0, 1.0)
        
        # Factor in threat type
        type_multiplier = 1.0
        if self.threat_type in ['apt', 'zero_day']:
            type_multiplier = 1.5
        elif self.threat_type in ['ransomware', 'trojan']:
            type_multiplier = 1.2
        
        return base_score * (1 + indicator_factor) * type_multiplier


class ThreatIntelligenceAdapter:
    """Adapts threat intelligence feeds to ThreatAntigens"""
    
    def __init__(self):
        self.known_threats = {}
        self.indicator_cache = {}
    
    def parse_stix_bundle(self, stix_data: Dict) -> List[ThreatAntigen]:
        """Parse STIX threat intelligence bundle"""
        threats = []
        
        # Extract threat actors, malware, and indicators
        for obj in stix_data.get('objects', []):
            if obj['type'] == 'malware':
                threat = self._create_threat_from_malware(obj)
                if threat:
                    threats.append(threat)
            elif obj['type'] == 'intrusion-set':
                threat = self._create_threat_from_apt(obj)
                if threat:
                    threats.append(threat)
        
        return threats
    
    def _create_threat_from_malware(self, malware_obj: Dict) -> Optional[ThreatAntigen]:
        """Create ThreatAntigen from STIX malware object"""
        name = malware_obj.get('name', 'Unknown')
        malware_types = malware_obj.get('malware_types', ['unknown'])
        
        # Create mock indicators for demo
        indicators = []
        for i in range(5):
            indicator = ThreatIndicator(
                indicator_type='hash',
                value=f"mock_hash_{name}_{i}",
                confidence=0.8,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                associated_malware=[name]
            )
            indicators.append(indicator)
        
        return ThreatAntigen(
            threat_name=name,
            threat_type=malware_types[0],
            indicators=indicators,
            severity=0.7
        )
    
    def _create_threat_from_apt(self, apt_obj: Dict) -> Optional[ThreatAntigen]:
        """Create ThreatAntigen from APT group"""
        name = apt_obj.get('name', 'Unknown APT')
        
        # Create diverse indicators for APT
        indicators = []
        
        # C2 servers
        for i in range(3):
            indicators.append(ThreatIndicator(
                indicator_type='ip',
                value=f"10.{i}.{i}.{i}",
                confidence=0.9,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            ))
        
        # Domains
        for i in range(2):
            indicators.append(ThreatIndicator(
                indicator_type='domain',
                value=f"malicious{i}.example.com",
                confidence=0.85,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            ))
        
        return ThreatAntigen(
            threat_name=name,
            threat_type='apt',
            indicators=indicators,
            attack_vector='network',
            severity=0.9
        )
    
    def create_synthetic_threats(self, count: int = 10) -> List[ThreatAntigen]:
        """Generate synthetic threats for testing"""
        threat_types = ['malware', 'ransomware', 'trojan', 'ddos', 'phishing', 
                       'apt', 'zero_day', 'botnet']
        attack_vectors = ['network', 'email', 'web', 'usb', 'supply_chain']
        
        threats = []
        for i in range(count):
            threat_type = np.random.choice(threat_types)
            num_indicators = np.random.randint(3, 20)
            
            indicators = []
            for j in range(num_indicators):
                indicator_type = np.random.choice(['ip', 'domain', 'hash', 'pattern'])
                indicator = ThreatIndicator(
                    indicator_type=indicator_type,
                    value=f"{indicator_type}_{i}_{j}",
                    confidence=np.random.uniform(0.5, 1.0),
                    first_seen=datetime.now(),
                    last_seen=datetime.now()
                )
                indicators.append(indicator)
            
            threat = ThreatAntigen(
                threat_name=f"Threat_{threat_type}_{i}",
                threat_type=threat_type,
                indicators=indicators,
                attack_vector=np.random.choice(attack_vectors),
                severity=np.random.uniform(0.3, 1.0)
            )
            threats.append(threat)
        
        return threats