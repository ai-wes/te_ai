

# ============================================================================
# BASE DIFFUSION DREAMER
# ============================================================================

class DiffusionDreamer(nn.Module):
    """Base diffusion model for dreaming antigens"""
    
    def __init__(self, feature_dim=cfg.feature_dim, hidden_dim=cfg.hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Noise schedule - dynamically sized to prevent OOB
        self.max_steps = 100
        self.noise_schedule = torch.linspace(1e-4, 0.02, self.max_steps)
        
    def add_noise(self, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t"""
        # Clamp t to valid range
        t = min(t, len(self.noise_schedule) - 1)
        alpha = 1 - self.noise_schedule[t]
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
        return noisy_x, noise
    
    def denoise(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Predict noise at timestep t"""
        t_embed = torch.tensor([t / 100.0], device=x.device).expand(x.shape[0], 1)
        x_with_t = torch.cat([x, t_embed], dim=-1)
        return self.denoise_net(x_with_t)
    
    def generate_dream_antigen(self, real_antigen: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generate dream antigen through reverse diffusion"""
        # Ensure steps doesn't exceed noise schedule length
        steps = min(steps, len(self.noise_schedule))
        x = torch.randn_like(real_antigen)
        
        for t in reversed(range(steps)):
            predicted_noise = self.denoise(x, t)
            
            alpha = 1 - self.noise_schedule[t]
            alpha_prev = 1 - self.noise_schedule[t-1] if t > 0 else 1.0
            
            # Reverse diffusion step
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            
            # Add noise for non-final steps
            if t > 0:
                sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                x += sigma * torch.randn_like(x)
        
        return x

# ============================================================================
# QUANTUM DIFFUSION DREAMER
# ============================================================================

class QuantumDiffusionDreamer(DiffusionDreamer):
    """Quantum-superposed diffusion for dreaming antigens in parallel realities."""
    
    def __init__(self, feature_dim=cfg.feature_dim, hidden_dim=cfg.hidden_dim):
        super().__init__(feature_dim, hidden_dim)
        
        # Create quantum gene for superposed denoising
        self.quantum_denoise = QuantumGeneModule('D', 42)
        self.quantum_denoise.to(cfg.device)
        
        # Quantum-aware denoising networks for each basis state
        self.denoise_0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.denoise_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Different activation for basis 1
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Interference network
        self.interference_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Entanglement tracker
        self.entangled_genes = []
        
    def superposed_denoise(self, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Denoise in quantum superposition"""
        # Create mock graph structure for gene forward pass
        num_nodes = x.shape[0]
        edge_index = torch.stack([
            torch.arange(num_nodes, device=x.device),
            torch.arange(num_nodes, device=x.device)
        ])  # Self-loops
        
        # Forward through quantum gene
        quantum_features = self.quantum_denoise(x, edge_index)
        
        # Get quantum probabilities
        prob_0, prob_1 = self.quantum_denoise.compute_probabilities()
        
        # Denoise in each basis
        denoised_0 = self.denoise_0(quantum_features)
        denoised_1 = self.denoise_1(quantum_features)
        
        # Compute interference term
        interference_input = torch.cat([denoised_0, denoised_1], dim=-1)
        interference = self.interference_net(interference_input)
        
        return denoised_0, denoised_1, interference
    
    def generate_dream_antigen(self, real_antigen: torch.Tensor, steps: int = 50, 
                             stress: float = 0.0, quantum_noise: float = 0.1) -> torch.Tensor:
        """
        Generate dream antigens using quantum diffusion.
        Higher stress causes faster decoherence (collapse to classical).
        """
        # Initialize in noise
        x = torch.randn_like(real_antigen)
        
        # Set decoherence rate based on stress
        self.quantum_denoise.decoherence_rate.data = torch.tensor(0.1 + stress * 0.5)
        
        # Track quantum evolution
        quantum_history = []
        
        for t in reversed(range(steps)):
            # Get superposed denoising
            denoised_0, denoised_1, interference = self.superposed_denoise(x, t)
            
            # Get current quantum state
            prob_0, prob_1 = self.quantum_denoise.compute_probabilities()
            quantum_interference = self.quantum_denoise.compute_interference(prob_0, prob_1)
            
            # Combine quantum paths
            denoised_super = (
                torch.sqrt(prob_0) * denoised_0 +
                torch.sqrt(prob_1) * denoised_1 +
                quantum_interference * interference
            )
            
            # Reverse diffusion step with safe indexing
            t_safe = min(t, len(self.noise_schedule) - 1)
            t_prev_safe = min(t-1, len(self.noise_schedule) - 1) if t > 0 else 0
            
            alpha = 1 - self.noise_schedule[t_safe]
            alpha_prev = 1 - self.noise_schedule[t_prev_safe] if t > 0 else 1.0
            
            # Ensure we don't divide by zero
            alpha = max(alpha, 1e-8)
            sqrt_one_minus_alpha = torch.sqrt(max(1 - alpha, 1e-8))
            
            x = (x - (1 - alpha) / sqrt_one_minus_alpha * denoised_super) / torch.sqrt(alpha)
            
            # Add noise with quantum fluctuations
            if t > 0:
                sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                quantum_fluctuation = quantum_noise * quantum_interference.abs()
                x += (sigma + quantum_fluctuation) * torch.randn_like(x)
            
            # Apply decoherence
            self.quantum_denoise.apply_decoherence()
            
            # Record quantum state
            quantum_history.append({
                'prob_0': prob_0.item(),
                'prob_1': prob_1.item(),
                'interference': quantum_interference.item()
            })
        
        # Final measurement collapses the quantum state
        outcome, probability = self.quantum_denoise.measure_quantum_state()
        
        # Blend based on measurement outcome
        if outcome == 0:
            # Collapsed to basis 0 - blend with reality
            dream_antigen = 0.7 * x + 0.3 * real_antigen
        else:
            # Collapsed to basis 1 - pure dream
            dream_antigen = x
        
        # Add metadata
        dream_antigen.quantum_history = quantum_history
        dream_antigen.measurement_outcome = outcome
        dream_antigen.collapse_probability = probability
        
        return dream_antigen
    
    def entangle_with_genes(self, genes: List['QuantumGeneModule']):
        """Create entanglement between dreamer and quantum genes"""
        for gene in genes:
            if isinstance(gene, QuantumGeneModule):
                self.quantum_denoise.entangle_with(gene)
                self.entangled_genes.append(gene)
    
    def dream_multiple_realities(self, real_antigens: List[torch.Tensor], 
                                num_realities: int = 5) -> List[torch.Tensor]:
        """Dream multiple parallel realities simultaneously"""
        dreams = []
        
        for i in range(num_realities):
            # Each reality has different quantum parameters
            with torch.no_grad():
                # Randomize quantum state
                theta = np.random.uniform(0, np.pi/2)
                self.quantum_denoise.alpha_amplitude.data = torch.cos(torch.tensor(theta))
                self.quantum_denoise.beta_amplitude.data = torch.sin(torch.tensor(theta))
                # Set random phase using sin/cos
                random_phase = np.random.uniform(-np.pi, np.pi)
                self.quantum_denoise.phase_sin.data = torch.sin(torch.tensor(random_phase))
                self.quantum_denoise.phase_cos.data = torch.cos(torch.tensor(random_phase))
                self.quantum_denoise._normalize_phase_components()
            
            # Generate dream in this reality
            antigen_idx = i % len(real_antigens)
            dream = self.generate_dream_antigen(
                real_antigens[antigen_idx],
                stress=i / num_realities,  # Increasing stress across realities
                quantum_noise=0.05 * (i + 1)
            )
            dreams.append(dream)
        
        return dreams

# ============================================================================
# ENHANCED QUANTUM DREAM CONSOLIDATION ENGINE
# ============================================================================




class DreamConsolidationEngine(nn.Module):
    """Complete dream-based learning system"""
    
    def __init__(self, input_dim: int = cfg.hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Dream generation network (VAE-style)
        self.dream_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2)
        )
        
        # Reparameterization for VAE
        self.mu_layer = nn.Linear(input_dim * 2, input_dim)
        self.logvar_layer = nn.Linear(input_dim * 2, input_dim)
        
        # Dream decoder
        self.dream_decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Nightmare generator (adversarial component)
        self.nightmare_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Dream critic (evaluates dream quality)
        self.dream_critic = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        
        # Memory systems
        self.episodic_memory = DreamMemory(capacity=10000)
        self.semantic_memory = DreamMemory(capacity=5000)
        
        # Consolidation networks
        self.consolidation_gru = nn.GRU(
            input_dim, input_dim, 
            num_layers=3, batch_first=True, dropout=0.1
        )
        
        self.consolidation_attention = nn.MultiheadAttention(
            input_dim, num_heads=8, batch_first=True
        )
        
        # Meta-learning components
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh()
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate_dream_batch(self, num_dreams: int) -> Tuple[torch.Tensor, Dict]:
        """Generate batch of dream experiences"""
        # Sample from episodic memory
        memories = self.episodic_memory.sample_batch(num_dreams * 2)
        
        if len(memories) < 2:
            return None, {}
        
        dream_states = []
        dream_metadata = {
            'vae_loss': [],
            'diversity': [],
            'novelty': []
        }
        
        for i in range(num_dreams):
            # Encode memory
            memory = random.choice(memories)
            state = memory['state'].to(cfg.device).unsqueeze(0)
            
            encoded = self.dream_encoder(state)
            mu = self.mu_layer(encoded)
            logvar = self.logvar_layer(encoded)
            
            # Generate dream variation
            z = self.reparameterize(mu, logvar)
            dream_state = self.dream_decoder(z)
            
            # VAE loss for quality monitoring
            recon_loss = F.mse_loss(dream_state, state)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + 0.01 * kl_loss
            
            dream_metadata['vae_loss'].append(vae_loss.item())
            
            # Add controlled noise for diversity
            if i % 3 == 0:  # Every third dream is a nightmare
                nightmare = self.nightmare_generator(dream_state)
                dream_state = dream_state + cfg.nightmare_adversarial_strength * nightmare
            
            dream_states.append(dream_state)
        
        if dream_states:
            dream_batch = torch.cat(dream_states, dim=0)
            
            # Compute diversity metrics
            dream_numpy = dream_batch.detach().cpu().numpy()
            pairwise_distances = np.linalg.norm(
                dream_numpy[:, np.newaxis] - dream_numpy[np.newaxis, :], 
                axis=2
            )
            dream_metadata['diversity'] = pairwise_distances.mean()
            
            # Compute novelty vs memories
            # Use min to avoid size mismatch when we have fewer memories than dreams
            num_memories_to_compare = min(len(memories), num_dreams, dream_batch.size(0))
            if num_memories_to_compare > 0:
                memory_states = torch.stack([m['state'] for m in memories[:num_memories_to_compare]]).to(cfg.device)
                dream_states_subset = dream_batch[:num_memories_to_compare]
                novelty = torch.norm(dream_states_subset - memory_states, dim=1).mean()
            else:
                novelty = torch.tensor(0.0)
            dream_metadata['novelty'] = novelty.item()
            
            return dream_batch, dream_metadata
        
        return None, dream_metadata
    
    def consolidate_learning(self, dream_batch: torch.Tensor, 
                           gene_states: List[torch.Tensor]) -> torch.Tensor:
        """Consolidate dream experiences into improved parameters"""
        if len(gene_states) == 0:
            return None
        
        # Stack gene states
        gene_tensor = torch.stack(gene_states).to(cfg.device)
        
        # Process dreams through GRU
        dream_hidden, _ = self.consolidation_gru(dream_batch.unsqueeze(0))
        dream_repr = dream_hidden.mean(dim=1)
        
        # Attention between dreams and current genes
        attended, attention_weights = self.consolidation_attention(
            gene_tensor.unsqueeze(0),
            dream_batch.unsqueeze(0),
            dream_batch.unsqueeze(0)
        )
        
        # Meta-learning: learn how to learn from dreams
        combined = torch.cat([
            gene_tensor.mean(dim=0),
            dream_repr.squeeze(0),
            attended.squeeze(0).mean(dim=0)
        ])
        
        meta_update = self.meta_learner(combined)
        
        return meta_update, attention_weights
    
    def evaluate_dream_quality(self, dream_batch: torch.Tensor, 
                             real_batch: torch.Tensor) -> float:
        """Evaluate quality and usefulness of dreams"""
        combined = torch.cat([dream_batch, real_batch], dim=1)
        quality_scores = self.dream_critic(combined)
        return quality_scores.mean().item()
    
# In the DreamConsolidationEngine class:

    def dream_phase(self, population: Dict[str, Any], num_cycles: int = 5):
        """
        Complete and optimized dream consolidation phase.
        
        Optimization:
        - Pre-computes a list of all eligible cells and their gene states in a single pass.
        - Avoids nested loops and repeated checks inside the main consolidation cycle.
        - Uses torch.no_grad() to prevent unnecessary gradient tracking.
        """
        print(f"\n💤 Dream Consolidation Phase ({num_cycles} cycles)")
        
        # --- OPTIMIZATION: Pre-computation Step ---
        # In a single pass, identify all cells eligible for consolidation and
        # extract their gene states. This is much faster than doing it repeatedly.
        
        eligible_cells_for_dreaming = []
        with torch.no_grad(): # No gradients needed for state extraction
            all_cells = list(population.values())
            for cell in all_cells:
                if not hasattr(cell, 'genes'):
                    continue

                # Extract states of active genes for this cell
                gene_states = [
                    gene.output_projection[0].weight.data.mean(dim=0)
                    for gene in cell.genes
                    if gene.is_active and hasattr(gene, 'output_projection')
                ]
                
                # A cell is eligible only if it has at least two active genes
                if len(gene_states) >= 2:
                    eligible_cells_for_dreaming.append({
                        'cell_obj': cell,
                        'gene_states': gene_states
                    })
        
        if not eligible_cells_for_dreaming:
            print("  No cells eligible for dream consolidation.")
            return
        # --- END OPTIMIZATION ---

        # Main consolidation loop
        for cycle in range(num_cycles):
            cycle_start = time.time()
            
            # Generate a batch of dream experiences
            dream_batch, dream_meta = self.generate_dream_batch(
                cfg.memory_replay_batch_size
            )
            
            if dream_batch is None:
                print("  Skipping dream cycle (not enough memories).")
                continue
            
            consolidation_count = 0
            total_improvement = 0.0
            
            # Process a random subset of the eligible cells for efficiency
            # This avoids processing the entire population every cycle
            cells_to_process = random.sample(
                eligible_cells_for_dreaming, 
                min(len(eligible_cells_for_dreaming), 100) # Process up to 100 cells per cycle
            )

            for cell_data in cells_to_process:
                cell = cell_data['cell_obj']
                gene_states = cell_data['gene_states']
                
                # Consolidate learning using the pre-computed gene states
                meta_update, attention = self.consolidate_learning(
                    dream_batch, gene_states
                )
                
                if meta_update is not None:
                    # Apply the consolidated learning update to the cell's genes
                    with torch.no_grad():
                        for i, gene in enumerate(cell.genes):
                            # Ensure we only update genes that contributed to the state
                            if gene.is_active and i < len(gene_states):
                                # Determine update strength using attention weights
                                if attention is not None and i < attention.shape[-1]:
                                    update_strength = attention[0, i, :].mean().item()
                                else:
                                    update_strength = 0.1
                                
                                # Modulate update by epigenetic state (less accessible genes change less)
                                update_strength *= (1.0 - gene.chromatin_accessibility)
                                
                                # Apply the update to all parameters of the gene
                                for param in gene.parameters():
                                    param.data += update_strength * torch.randn_like(param) * \
                                                  meta_update.norm().item() * 0.01
                    
                    consolidation_count += 1
                    total_improvement += meta_update.norm().item()
            
            # Log the results of the consolidation cycle
            cycle_time = time.time() - cycle_start
            avg_improvement = total_improvement / max(consolidation_count, 1)
            print(f"  Cycle {cycle+1}: {consolidation_count} cells consolidated, "
                  f"avg improvement: {avg_improvement:.4f}, "
                  f"time: {cycle_time:.2f}s")
            
            if dream_meta and 'vae_loss' in dream_meta and dream_meta['vae_loss']:
                print(f"    Dream quality - VAE loss: {np.mean(dream_meta['vae_loss']):.4f}, "
                      f"diversity: {dream_meta.get('diversity', 0):.4f}, "
                      f"novelty: {dream_meta.get('novelty', 0):.4f}")
                
                
                










class QuantumDreamConsolidationEngine(DreamConsolidationEngine):
    """Enhanced dream engine with quantum dreaming capabilities"""
    
    def __init__(self, input_dim: int = cfg.hidden_dim):
        super().__init__(input_dim)
        
        # Add quantum dreamer
        self.quantum_dreamer = QuantumDiffusionDreamer()
        
        # Quantum memory for storing superposed states
        self.quantum_memory = deque(maxlen=1000)
        
        # Reality fusion network
        self.reality_fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def quantum_dream_phase(self, population: Dict, antigens: List[torch.Tensor], 
                          num_cycles: int = 5):
        """Execute quantum dream consolidation with parallel realities"""
        print(f"\n🌌 Quantum Dream Phase ({num_cycles} cycles)")
        
        # Find all quantum genes in population
        quantum_genes = []
        for cell in population.values():
            for gene in cell.genes:
                if isinstance(gene, QuantumGeneModule) and gene.is_active:
                    quantum_genes.append(gene)
        
        if quantum_genes:
            print(f"   Found {len(quantum_genes)} quantum genes for entanglement")
            # Entangle dreamer with population's quantum genes
            self.quantum_dreamer.entangle_with_genes(quantum_genes[:10])  # Limit for performance
        
        for cycle in range(num_cycles):
            cycle_start = time.time()
            
            # Generate dreams in multiple realities
            dream_antigens = self.quantum_dreamer.dream_multiple_realities(
                antigens, 
                num_realities=3 + cycle  # Increase realities as we go
            )
            
            # Process dreams through population
            consolidation_count = 0
            total_quantum_improvement = 0.0
            
            for i, dream_antigen in enumerate(dream_antigens):
                # Select random cells for this reality
                cells_in_reality = random.sample(
                    list(population.values()), 
                    min(20, len(population))
                )
                
                for cell in cells_in_reality:
                    # Evaluate cell response to dream antigen
                    mock_batch = self._create_antigen_batch([dream_antigen])
                    affinity, representation, _ = cell(mock_batch)
                    
                    # Quantum-enhanced learning
                    if hasattr(dream_antigen, 'measurement_outcome'):
                        # Adjust learning based on quantum collapse
                        quantum_factor = dream_antigen.collapse_probability
                        learning_rate = cfg.dream_learning_rate * (1 + quantum_factor)
                        
                        # Update cell based on dream response
                        self._apply_quantum_learning(
                            cell, 
                            affinity, 
                            representation,
                            dream_antigen.measurement_outcome,
                            learning_rate
                        )
                        
                        consolidation_count += 1
                        total_quantum_improvement += affinity.mean().item()
            
            # Log cycle results
            cycle_time = time.time() - cycle_start
            avg_improvement = total_quantum_improvement / max(consolidation_count, 1)
            
            print(f"   Cycle {cycle+1}: {consolidation_count} quantum consolidations, "
                  f"avg improvement: {avg_improvement:.4f}, "
                  f"realities: {len(dream_antigens)}, "
                  f"time: {cycle_time:.2f}s")
            
            # Store quantum states in memory
            for gene in quantum_genes[:5]:  # Store a few for efficiency
                self.quantum_memory.append({
                    'cycle': cycle,
                    'prob_0': gene.compute_probabilities()[0].item(),
                    'prob_1': gene.compute_probabilities()[1].item(),
                    'coherence': gene.coherence_steps
                })
    
    def _create_antigen_batch(self, antigens: List[torch.Tensor]):
        """Create a batch from antigen tensors"""
        from torch_geometric.data import Data, Batch
        
        data_list = []
        for antigen in antigens:
            # Create simple graph structure
            num_nodes = antigen.shape[0]
            edge_index = torch.stack([
                torch.arange(num_nodes, device=antigen.device),
                torch.arange(num_nodes, device=antigen.device)
            ])
            
            data = Data(x=antigen, edge_index=edge_index)
            data_list.append(data)
        
        return Batch.from_data_list(data_list)
    
    def _apply_quantum_learning(self, cell, affinity, representation, 
                               quantum_outcome, learning_rate):
        """Apply quantum-enhanced learning with gradient consolidation"""
        # First, apply gradient-based learning to consolidate promising directions
        optim = torch.optim.Adam(cell.parameters(), lr=learning_rate)
        optim.zero_grad()
        
        # Maximize affinity through gradient ascent
        # Use the existing affinity tensor but handle gradient issues
        try:
            loss = -affinity.mean()
            if loss.requires_grad:
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(cell.parameters(), 1.0)
                optim.step()
        except RuntimeError:
            # If gradient computation fails, skip gradient step
            pass
        
        # Clear gradients to prevent accumulation
        optim.zero_grad()
        
        # Then add quantum noise for diversity (with no_grad to avoid interfering with gradients)
        with torch.no_grad():
            # Update genes based on quantum outcome
            for gene in cell.genes:
                if gene.is_active:
                    # Get the target device from the gene's parameters
                    try:
                        target_device = next(gene.parameters()).device
                    except StopIteration:
                        # Skip this gene if it has no parameters
                        continue
                    
                    # Ensure the affinity tensor is on the same device as the gene's parameters
                    affinity_on_device = affinity.to(target_device)

                    # Basis-dependent update - reduced noise magnitude since gradient step already applied
                    if quantum_outcome == 0:
                        # Conservative update (reality-anchored)
                        for param in gene.parameters():
                            noise = torch.randn_like(param)
                            param.data += learning_rate * 0.1 * noise * affinity_on_device.to(param.device)
                    else:
                        # Explorative update (pure dream)
                        for param in gene.parameters():
                            noise = torch.randn_like(param)
                            param.data += learning_rate * 0.5 * noise * affinity_on_device.to(param.device)
                            
    # Visualize quantum dream statistics
    # This method will print statistics about the quantum dreams stored in memory.    
    def visualize_quantum_dreams(self):
        """Visualize quantum dream statistics"""
        if not self.quantum_memory:
            return
        
        recent_states = list(self.quantum_memory)[-100:]
        
        # Extract statistics
        prob_0_history = [s['prob_0'] for s in recent_states]
        prob_1_history = [s['prob_1'] for s in recent_states]
        coherence_history = [s['coherence'] for s in recent_states]
        
        print("\n📊 Quantum Dream Statistics:")
        print(f"   Average |0⟩ probability: {np.mean(prob_0_history):.3f}")
        print(f"   Average |1⟩ probability: {np.mean(prob_1_history):.3f}")
        print(f"   Average coherence time: {np.mean(coherence_history):.1f} steps")
        print(f"   Quantum memory size: {len(self.quantum_memory)} states")