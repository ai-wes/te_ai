GRAND_TAPESTRY = CausalTapestry()
GRAND_TAPESTRY.load_tapestry("tapestry_efficacy_toxicity.graphml")
GRAND_TAPESTRY.merge_tapestry("tapestry_adme_synth.graphml")
GRAND_TAPESTRY.merge_tapestry("tapestry_novelty_speed.graphml")