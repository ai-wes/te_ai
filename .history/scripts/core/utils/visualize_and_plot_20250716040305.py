








# ============================================================================
# PLOT, VISUALIZE, AND ANALYZE FUNCTIONS
# ============================================================================





def visualize_production_state(center: ProductionGerminalCenter, epoch: int):
    """Create comprehensive visualization of current state"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Gene topology with depth
    ax1 = plt.subplot(3, 3, 1)
    visualize_gene_topology_3d(center, ax1)
    
    # 2. Fitness landscape
    ax2 = plt.subplot(3, 3, 2)
    plot_fitness_landscape(center, ax2)
    
    # 3. Phase space
    ax3 = plt.subplot(3, 3, 3)
    plot_phase_space(center, ax3)
    
    # 4. Diversity metrics
    ax4 = plt.subplot(3, 3, 4)
    plot_diversity_metrics(center, ax4)
    
    # 5. Gene expression heatmap
    ax5 = plt.subplot(3, 3, 5)
    plot_gene_expression_heatmap(center, ax5)
    
    # 6. Transposition events
    ax6 = plt.subplot(3, 3, 6)
    plot_transposition_timeline(center, ax6)
    
    # 7. Population structure
    ax7 = plt.subplot(3, 3, 7)
    plot_population_structure(center, ax7)
    
    # 8. Epigenetic landscape
    ax8 = plt.subplot(3, 3, 8)
    plot_epigenetic_landscape(center, ax8)
    
    # 9. Performance metrics
    ax9 = plt.subplot(3, 3, 9)
    plot_performance_summary(center, ax9)
    
    plt.suptitle(f'Transposable Element AI - Generation {center.generation}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(cfg.save_dir, f'state_gen_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_gene_topology_3d(center: ProductionGerminalCenter, ax):
    """3D visualization of gene arrangements"""
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(3, 3, 1, projection='3d')
    
    # Sample cells
    sample_cells = list(center.population.values())[:20]
    
    for i, cell in enumerate(sample_cells):
        for gene in cell.genes:
            if gene.is_active:
                x = gene.position
                y = i / len(sample_cells)
                z = gene.compute_depth().item()
                
                color = {'V': 'red', 'D': 'green', 'J': 'blue'}.get(gene.gene_type, 'gray')
                size = 50 * (1 + gene.fitness_contribution)
                
                ax.scatter(x, y, z, c=color, s=size, alpha=0.6)
    
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Cell Index')
    ax.set_zlabel('Neural Depth')
    ax.set_title('3D Gene Topology')

def plot_fitness_landscape(center: ProductionGerminalCenter, ax):
    """Plot fitness evolution with phase transitions"""
    if not center.fitness_landscape:
        return
    
    generations = [d['generation'] for d in center.fitness_landscape]
    mean_fitness = [d['mean_fitness'] for d in center.fitness_landscape]
    max_fitness = [d['max_fitness'] for d in center.fitness_landscape]
    
    ax.plot(generations, mean_fitness, 'b-', label='Mean', linewidth=2)
    ax.plot(generations, max_fitness, 'g--', label='Max', linewidth=2)
    
    # Mark phase transitions
    for transition in center.phase_detector.transition_history:
        gen = transition['metrics'].get('generation', 0)
        ax.axvline(x=gen, color='red', alpha=0.3, linestyle=':')
        ax.text(gen, ax.get_ylim()[1], transition['to_phase'][:4], 
               rotation=90, va='top', fontsize=8)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_phase_space(center: ProductionGerminalCenter, ax):
    """Plot phase space diagram"""
    phase_data = center.phase_detector.get_phase_diagram_data()
    
    if phase_data and 'autocorrelation' in phase_data and 'variance' in phase_data:
        ax.scatter(phase_data['autocorrelation'], phase_data['variance'],
                  c=phase_data['phase_colors'], s=50, alpha=0.6)
        
        # Add phase boundaries
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Autocorrelation')
        ax.set_ylabel('Variance')
        ax.set_title('Phase Space')
        
        # Add phase labels
        ax.text(0.2, 0.05, 'Stable', fontsize=10, ha='center')
        ax.text(0.9, 0.05, 'Critical', fontsize=10, ha='center')
        ax.text(0.9, 0.3, 'Bifurcation', fontsize=10, ha='center')

def plot_diversity_metrics(center: ProductionGerminalCenter, ax):
    """Plot diversity evolution"""
    if not center.diversity_metrics:
        return
    
    generations = [d['generation'] for d in center.diversity_metrics]
    shannon = [d.get('shannon_index', 0) for d in center.diversity_metrics]
    simpson = [d.get('simpson_index', 0) for d in center.diversity_metrics]
    
    ax.plot(generations, shannon, 'purple', label='Shannon', linewidth=2)
    ax.plot(generations, simpson, 'orange', label='Simpson', linewidth=2)
    ax.axhline(y=cfg.shannon_entropy_target, color='red', linestyle='--', 
              alpha=0.5, label='Target')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity Index')
    ax.set_title('Population Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_gene_expression_heatmap(center: ProductionGerminalCenter, ax):
    """Heatmap of gene expression patterns"""
    # Sample cells and genes
    sample_size = min(30, len(center.population))
    sample_cells = list(center.population.values())[:sample_size]
    
    expression_matrix = []
    
    for cell in sample_cells:
        cell_expression = []
        for gene in cell.genes[:10]:  # First 10 genes
            if gene.is_active:
                methylation_level = gene.methylation_state.mean().item()
                expression = gene.chromatin_accessibility * (1 - methylation_level)
                cell_expression.append(expression)
            else:
                cell_expression.append(0)
                
        # Pad to fixed size
        while len(cell_expression) < 10:
            cell_expression.append(0)
        
        expression_matrix.append(cell_expression)
    
    if expression_matrix:
        im = ax.imshow(expression_matrix, aspect='auto', cmap='RdYlBu_r')
        ax.set_xlabel('Gene Index')
        ax.set_ylabel('Cell Index')
        ax.set_title('Gene Expression Heatmap')
        plt.colorbar(im, ax=ax, fraction=0.046)

def plot_transposition_timeline(center: ProductionGerminalCenter, ax):
    """Timeline of transposition events"""
    if not center.transposition_events:
        return
    
    # Count events by type and generation
    event_counts = defaultdict(lambda: defaultdict(int))
    
    for event in center.transposition_events[-1000:]:  # Last 1000 events
        gen = event['generation']
        action = event['event']['action']
        event_counts[action][gen] += 1
    
    # Plot stacked area chart
    generations = sorted(set(g for counts in event_counts.values() for g in counts))
    
    jump_counts = [event_counts['jump'].get(g, 0) for g in generations]
    dup_counts = [event_counts['duplicate'].get(g, 0) for g in generations]
    inv_counts = [event_counts['invert'].get(g, 0) for g in generations]
    del_counts = [event_counts['delete'].get(g, 0) for g in generations]
    
    ax.stackplot(generations, jump_counts, dup_counts, inv_counts, del_counts,
                labels=['Jump', 'Duplicate', 'Invert', 'Delete'],
                colors=['blue', 'green', 'orange', 'red'],
                alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Transposition Events')
    ax.set_title('Transposition Timeline')
    ax.legend(loc='upper left')

def plot_population_structure(center: ProductionGerminalCenter, ax):
    """Population structure visualization"""
    # Compute population statistics
    fitness_values = []
    gene_counts = []
    lineage_depths = []
    
    for cell in center.population.values():
        if cell.fitness_history:
            fitness_values.append(cell.fitness_history[-1])
        else:
            fitness_values.append(0)
        
        gene_counts.append(len([g for g in cell.genes if g.is_active]))
        lineage_depths.append(len(cell.lineage))
    
    # Create 2D histogram
    if fitness_values and gene_counts:
        h = ax.hist2d(fitness_values, gene_counts, bins=20, cmap='YlOrRd')
        plt.colorbar(h[3], ax=ax)
        
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Active Gene Count')
        ax.set_title('Population Structure')

def plot_epigenetic_landscape(center: ProductionGerminalCenter, ax):
    """Epigenetic modification landscape"""
    methylation_levels = []
    histone_states = []
    
    # Sample genes
    for cell in list(center.population.values())[:50]:
        for gene in cell.genes:
            if gene.is_active:
                methylation = gene.methylation_state.mean().item()
                methylation_levels.append(methylation)
                
                # Compute histone state
                h3k4me3 = torch.sigmoid(gene.histone_modifications[0]).item()
                h3k27me3 = torch.sigmoid(gene.histone_modifications[1]).item()
                histone_state = h3k4me3 - h3k27me3  # Active - repressive
                histone_states.append(histone_state)
    
    if methylation_levels and histone_states:
        ax.scatter(methylation_levels, histone_states, alpha=0.5, s=30)
        ax.set_xlabel('Methylation Level')
        ax.set_ylabel('Histone State (Active - Repressive)')
        ax.set_title('Epigenetic Landscape')
        ax.grid(True, alpha=0.3)

def plot_performance_summary(center: ProductionGerminalCenter, ax):
    """Summary performance metrics"""
    ax.axis('off')
    
    # Compute summary statistics
    current_gen = center.generation
    
    if center.fitness_landscape:
        current_fitness = center.fitness_landscape[-1]['mean_fitness']
        max_fitness_ever = max(d['max_fitness'] for d in center.fitness_landscape)
    else:
        current_fitness = 0
        max_fitness_ever = 0
    
    total_transpositions = len(center.transposition_events)
    
    if center.diversity_metrics:
        current_diversity = center.diversity_metrics[-1]['shannon_index']
    else:
        current_diversity = 0
    
    current_phase = center.phase_detector.current_phase
    population_size = len(center.population)
    
    # Create text summary
    summary_text = f"""
    PERFORMANCE SUMMARY
    ==================
    
    Generation: {current_gen}
    Population Size: {population_size}
    
    Fitness:
      Current Mean: {current_fitness:.4f}
      Best Ever: {max_fitness_ever:.4f}
    
    Diversity:
      Shannon Index: {current_diversity:.4f}
      Phase State: {current_phase}
    
    Evolution:
      Total Transpositions: {total_transpositions}
      Stress Level: {center.current_stress:.3f}
    
    System Health:
      GPU Utilization: {get_gpu_utilization():.1f}%
      Memory Usage: {get_memory_usage():.1f}%
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')

def get_gpu_utilization():
    """Get current GPU utilization"""
    if torch.cuda.is_available():
        return torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0
    return 0.0

def get_memory_usage():
    """Get current memory usage"""
    import psutil
    return psutil.virtual_memory().percent

def final_analysis(center: ProductionGerminalCenter, total_time: float):
    """Comprehensive final analysis"""
    print(f"\n📊 Final Analysis:")
    print(f"   Total runtime: {total_time/3600:.2f} hours")
    print(f"   Generations completed: {center.generation}")
    print(f"   Final population size: {len(center.population)}")
    
    if center.fitness_landscape:
        final_fitness = center.fitness_landscape[-1]['mean_fitness']
        max_fitness = max(d['max_fitness'] for d in center.fitness_landscape)
        print(f"   Final mean fitness: {final_fitness:.4f}")
        print(f"   Best fitness achieved: {max_fitness:.4f}")
    
    print(f"\n🧬 Evolutionary Statistics:")
    print(f"   Total transposition events: {len(center.transposition_events)}")
    
    # Count event types
    event_types = defaultdict(int)
    for event in center.transposition_events:
        event_types[event['event']['action']] += 1
    
    for action, count in event_types.items():
        print(f"   - {action}: {count}")
    
    if center.diversity_metrics:
        final_diversity = center.diversity_metrics[-1]
        print(f"\n🌈 Final Diversity:")
        print(f"   Shannon Index: {final_diversity['shannon_index']:.4f}")
        print(f"   Gene Richness: {final_diversity['gene_richness']}")
    
    print(f"\n🔄 Phase Transitions:")
    print(f"   Total transitions: {len(center.phase_detector.transition_history)}")
    for transition in center.phase_detector.transition_history[-5:]:
        print(f"   - Gen {transition['metrics'].get('generation', 0)}: "
              f"{transition['from_phase']} → {transition['to_phase']}")
    
    # Save final results
    results_path = os.path.join(cfg.save_dir, 'final_results.json')
    results = {
        'config': cfg.__dict__,
        'runtime_hours': total_time / 3600,
        'generations': center.generation,
        'final_population_size': len(center.population),
        'fitness_landscape': center.fitness_landscape,
        'diversity_metrics': center.diversity_metrics,
        'phase_transitions': [
            {
                'generation': t['metrics'].get('generation', 0),
                'from_phase': t['from_phase'],
                'to_phase': t['to_phase']
            }
            for t in center.phase_detector.transition_history
        ],
        'event_counts': dict(event_types)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {results_path}")
    
    # Generate final visualizations
    print("\n🎨 Generating final visualizations...")
    visualize_production_state(center, center.generation)
    
    # Create summary plot
    create_summary_figure(center)
    
    print("\n✅ Simulation complete!")

def create_summary_figure(center: ProductionGerminalCenter):
    """Create comprehensive summary figure"""
    fig = plt.figure(figsize=(24, 16))
    
    # Main fitness plot
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    plot_fitness_landscape(center, ax1)
    
    # Phase diagram
    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    plot_phase_space(center, ax2)
    
    # Diversity
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    plot_diversity_metrics(center, ax3)
    
    # Transpositions
    ax4 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    plot_transposition_timeline(center, ax4)
    
    # Gene expression
    ax5 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    plot_gene_expression_heatmap(center, ax5)
    
    # Summary text
    ax6 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    plot_performance_summary(center, ax6)
    
    plt.suptitle('Transposable Element AI - Complete Evolution Summary', fontsize=20)
    plt.tight_layout()
    
    save_path = os.path.join(cfg.save_dir, 'evolution_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Summary figure saved to {save_path}")
