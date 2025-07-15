class GuidedStemGene(StemGeneModule):
    """Stem gene that learns from seed genes"""
    
    def __init__(self):
        super().__init__()
        
        # Learning from seeds
        self.observed_patterns = {}
        self.skill_memory = nn.Parameter(torch.zeros(10, cfg.hidden_dim))
        self.confidence_scores = {}
        
        # Attention mechanism for learning from multiple teachers
        self.teacher_attention = nn.MultiheadAttention(
            cfg.hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        
    def observe_seed_genes(self, population):
        """Learn from successful seed genes"""
        seed_genes = []
        seed_features = []
        seed_performance = []
        
        for cell in population.values():
            for gene in cell.genes:
                if hasattr(gene, 'is_seed') and gene.is_seed:
                    seed_genes.append(gene)
                    seed_features.append(gene.morphogen_field)
                    seed_performance.append(gene.fitness_contribution)
        
        if not seed_genes:
            return None
            
        # Learn successful patterns
        features = torch.stack(seed_features)
        performance = torch.tensor(seed_performance)
        
        # Weight by performance
        weights = F.softmax(performance / 0.1, dim=0)
        
        # Attend to successful patterns
        query = self.skill_memory.unsqueeze(0)
        keys = features.unsqueeze(0)
        values = features.unsqueeze(0)
        
        attended_skills, attention_weights = self.teacher_attention(
            query, keys, values
        )
        
        # Store learned patterns
        best_teacher_idx = torch.argmax(weights)
        best_teacher = seed_genes[best_teacher_idx]
        
        self.observed_patterns[best_teacher.gene_type] = {
            'signature': best_teacher.domain_signature,
            'performance': performance[best_teacher_idx].item(),
            'pattern': attended_skills.squeeze(0)
        }
        
        return best_teacher.gene_type, attention_weights
    
    def guided_differentiation(self, stress_level: float = 0.0):
        """Differentiate based on learned patterns"""
        
        # First, learn from environment
        learned_type, attention = self.observe_seed_genes(self.get_population())
        
        if learned_type and random.random() < 0.7:  # 70% chance to follow teacher
            # Partial differentiation toward successful pattern
            self.differentiation_state.data = torch.zeros(4)
            
            # Don't fully commit - maintain some plasticity
            differentiation_strength = min(0.8, self.confidence_scores.get(learned_type, 0.5))
            
            type_map = {'MD': 0, 'NA': 1, 'TP': 2, 'S': 3}
            if learned_type in type_map:
                self.differentiation_state.data[type_map[learned_type]] = differentiation_strength
                self.differentiation_state.data[3] = 1 - differentiation_strength  # Remain partly stem
                
                # Absorb some characteristics
                if learned_type in self.observed_patterns:
                    pattern = self.observed_patterns[learned_type]
                    self.skill_memory.data[0] = pattern['pattern'][0]  # Learn first skill
                    
                print(f"   📚 Stem gene learned {learned_type} pattern (confidence: {differentiation_strength:.2f})")
        else:
            # Explore novel combinations
            self.explore_novel_differentiation()
    
    def explore_novel_differentiation(self):
        """Create novel gene types by combining learned patterns"""
        if len(self.observed_patterns) >= 2:
            # Combine two successful patterns
            types = list(self.observed_patterns.keys())
            type1, type2 = random.sample(types, 2)
            
            # Create hybrid differentiation
            self.differentiation_state.data = torch.zeros(4)
            type_map = {'MD': 0, 'NA': 1, 'TP': 2, 'S': 3}
            
            # 40% each from teachers, 20% stem
            self.differentiation_state.data[type_map[type1]] = 0.4
            self.differentiation_state.data[type_map[type2]] = 0.4
            self.differentiation_state.data[3] = 0.2
            
            print(f"   🧪 Stem gene exploring hybrid: {type1}+{type2}")
