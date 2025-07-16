class TransposableNeuralCell {
  constructor() {
    // Define the new architectural zones based on the diagram
    this.zones = {
      memoryEliteZone: {
        // First immediate sphere surrounding central cylinder
        center: new THREE.Vector3(0, 0, 0),
        radius: 50,
        color: 0x44aaaa, // Cyan
        description: "Memory Elite Zone - stable high performers",
      },
      lightZone: {
        // Gold helix cylinder - selection zone
        center: new THREE.Vector3(0, 0, 0),
        radius: 25, // Same as central cylinder
        height: 260, // Same as central cylinder
        color: 0xffd700, // Gold
        description: "Selection Light Zone - gold helix cylinder",
      },
      darkZone: {
        // Mutation zone - cosmic abyss with flashes
        center: new THREE.Vector3(0, 0, 0),
        innerRadius: 80, // Outside mantle zone
        outerRadius: 150,
        height: 100,
        color: 0x000033, // Dark cosmic blue
        flashColor: 0xff00ff, // Bright pink flashes
        description: "Mutation zone - dark cosmic abyss",
      },
      quantumDisk: {
        // Top and bottom quantum disks
        topCenter: new THREE.Vector3(0, 130, 0),
        bottomCenter: new THREE.Vector3(0, -130, 0),
        radius: 100,
        thickness: 10,
        color: 0xff00ff, // Magenta/Purple
        description: "4D Quantum disk with central connection",
      },
      mantleZone: {
        // Outer sphere surrounding memory elite zone
        center: new THREE.Vector3(0, 0, 0),
        radius: 80, // Outside of memory elite zone (50)
        color: 0x333366, // Dark blue-gray
        description: "Mantle/Transitional zone",
      },
      centralCylinder: {
        // Central connecting cylinder
        center: new THREE.Vector3(0, 0, 0),
        radius: 25,
        height: 260,
        color: 0x888844, // Golden
        description: "Central quantum connection cylinder",
      },
    };

    // Network connections for HGT
    this.plasmidNetwork = new Map();
    this.lineageTree = new Map();
  }

  // Zone positioning methods for the hierarchical biological layout

  memoryEliteZonePosition(cellData, index) {
    // Elite cells in sphere surrounding central cylinder
    const zone = this.zones.memoryEliteZone;
    const fitness = cellData.fitness || 0.5;

    // Use fibonacci sphere distribution for even spacing
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const y = 1 - (index / 50) * 2; // Maps to [-1, 1]
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = goldenAngle * index;

    // Higher fitness cells get closer to center
    const radiusScale = 0.5 + fitness * 0.5;
    const x = Math.cos(theta) * radiusAtY * zone.radius * radiusScale;
    const z = Math.sin(theta) * radiusAtY * zone.radius * radiusScale;

    return new THREE.Vector3(x, y * zone.radius * radiusScale, z);
  }

  lightZonePosition(cellData, index, population) {
    // Gold helix cylinder - selection zone
    const zone = this.zones.lightZone;
    const fitness = cellData.fitness || 0.5;

    // Create helical pattern within the central cylinder
    const totalCells = population.length;
    const t = (index / totalCells) * 6; // 6 full turns
    const angle = t * Math.PI * 2;

    // Helix parameters - inside the central cylinder
    const helixRadius = zone.radius * (0.6 + fitness * 0.3); // Fitter cells spiral closer to center
    const helixHeight = zone.height * (t / 6); // Vertical progression

    const x = Math.cos(angle) * helixRadius;
    const z = Math.sin(angle) * helixRadius;
    const y = zone.center.y - zone.height / 2 + helixHeight;

    return new THREE.Vector3(x, y, z);
  }

  darkZonePosition(cellData, index) {
    // Mutation zone - cosmic abyss with concentric layers
    const zone = this.zones.darkZone;
    const mutationRate = this.calculateMutationRate(cellData);

    // Create layers in the dark zone
    const layers = 3;
    const layerIndex = index % layers;
    const angleStep = (Math.PI * 2) / (layers * 10);
    const angle = index * angleStep;

    // Distance from center varies by layer and mutation intensity
    const baseRadius =
      zone.innerRadius +
      (zone.outerRadius - zone.innerRadius) * (layerIndex / layers);
    const radiusVariation = mutationRate * 20; // More mutation = more chaotic positioning
    const radius = baseRadius + Math.sin(angle * 3) * radiusVariation;

    // Height varies creating a cosmic cloud effect
    const height = (Math.random() - 0.5) * zone.height;

    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = zone.center.y + height;

    return new THREE.Vector3(x, y, z);
  }

  quantumDiskPosition(cellData, index) {
    // 4D quantum disks - top and bottom
    const zone = this.zones.quantumDisk;
    const isTopDisk = index % 2 === 0;
    const center = isTopDisk ? zone.topCenter : zone.bottomCenter;

    // Spiral pattern on the disk
    const spiralTurns = 5;
    const t = (index / 50) % 1; // Normalize to 0-1
    const angle = t * spiralTurns * Math.PI * 2;
    const radius = t * zone.radius;

    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = center.y + (Math.random() - 0.5) * zone.thickness;

    return new THREE.Vector3(center.x + x, y, center.z + z);
  }

  mantleZonePosition(cellData, index) {
    // Outer transitional sphere - fibonacci sphere distribution
    const zone = this.zones.mantleZone;
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));

    const y = 1 - (index / 100) * 2; // Maps to [-1, 1]
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = goldenAngle * index;

    const x = Math.cos(theta) * radiusAtY * zone.radius;
    const z = Math.sin(theta) * radiusAtY * zone.radius;

    return new THREE.Vector3(x, y * zone.radius, z);
  }

  centralCylinderPosition(cellData, index) {
    // Central connecting cylinder - vertical arrangement
    const zone = this.zones.centralCylinder;
    const height = (index / 20) * zone.height - zone.height / 2;
    const angle = (index / 5) * Math.PI * 2;

    const x = Math.cos(angle) * zone.radius * 0.8;
    const z = Math.sin(angle) * zone.radius * 0.8;
    const y = zone.center.y + height;

    return new THREE.Vector3(x, y, z);
  }

  calculateMutationRate(cellData) {
    // Calculate mutation intensity based on cell properties
    const stressLevel = cellData.stress_level || 0;
    const generation = cellData.generation || 0;
    const fitness = cellData.fitness || 0.5;

    // Higher stress and later generations increase mutation rate
    // Lower fitness also increases mutation tendency
    return Math.min(
      stressLevel * 0.5 + generation * 0.1 + (1 - fitness) * 0.3,
      1.0
    );
  }

  // Zone assignment logic based on cell characteristics
  determineZone(cellData) {
    const fitness = cellData.fitness || 0.5;
    const generation = cellData.generation || 0;
    const stressLevel = cellData.stress_level || 0;
    const hasQuantumGenes = cellData.genes?.some((g) => g.is_quantum) || false;

    // Elite memory cells (top 10% fitness, stable across generations)
    if (fitness > 0.9 && generation > 5) {
      return "memoryEliteZone";
    }

    // Quantum cells go to quantum disks
    if (hasQuantumGenes) {
      return "quantumDisk";
    }

    // High stress/mutation cells go to dark zone
    if (stressLevel > 0.6 || this.calculateMutationRate(cellData) > 0.7) {
      return "darkZone";
    }

    // Medium fitness cells in selection zone (light zone helix)
    if (fitness > 0.4 && fitness < 0.9) {
      return "lightZone";
    }

    // Central cylinder for special cells (stem cells, highly connected)
    const hasStemGenes =
      cellData.genes?.some((g) => g.gene_type === "S") || false;
    if (hasStemGenes) {
      return "centralCylinder";
    }

    // Everything else goes to mantle zone
    return "mantleZone";
  }

  // Create zone visualizations
  createZoneVisualizations(scene) {
    this.zoneGroup = new THREE.Group();

    // Memory Elite Zone - glowing cylinder at top
    this.createMemoryEliteZone(scene);

    // Light Zone - helix wireframe
    this.createLightZoneHelix(scene);

    // Dark Zone - cosmic abyss with particle effects
    this.createDarkZoneAbyss(scene);

    // Quantum Disks - top and bottom
    this.createQuantumDisks(scene);

    // Mantle Zone - outer sphere
    this.createMantleZone(scene);

    // Central Cylinder - connecting core
    this.createCentralCylinder(scene);

    // Add all zones to scene
    scene.add(this.zoneGroup);
  }

  createMemoryEliteZone(scene) {
    const zone = this.zones.memoryEliteZone;

    // Glowing sphere surrounding central cylinder
    const sphereGeometry = new THREE.SphereGeometry(zone.radius, 32, 16);
    const sphereMaterial = new THREE.MeshPhysicalMaterial({
      color: zone.color,
      emissive: zone.color,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.2,
      metalness: 0.8,
      roughness: 0.2,
    });

    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.copy(zone.center);
    sphere.frustumCulled = false;
    this.zoneGroup.add(sphere);
  }

  createLightZoneHelix(scene) {
    const zone = this.zones.lightZone;

    // Create helix wireframe
    const helixPoints = [];
    const turns = 4;
    const segments = 200;

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const angle = t * turns * Math.PI * 2;
      const y = zone.center.y - zone.height / 2 + t * zone.height;
      const x = Math.cos(angle) * zone.radius;
      const z = Math.sin(angle) * zone.radius;

      helixPoints.push(new THREE.Vector3(x, y, z));
    }

    const helixGeometry = new THREE.BufferGeometry().setFromPoints(helixPoints);
    const helixMaterial = new THREE.LineBasicMaterial({
      color: zone.color,
      transparent: true,
      opacity: 0.8,
      linewidth: 3,
    });

    const helix = new THREE.Line(helixGeometry, helixMaterial);
    helix.frustumCulled = false;
    this.zoneGroup.add(helix);
  }

  createDarkZoneAbyss(scene) {
    const zone = this.zones.darkZone;

    // Create concentric rings for the abyss
    for (let i = 0; i < 3; i++) {
      const radius =
        zone.innerRadius + (zone.outerRadius - zone.innerRadius) * (i / 2);
      const ringGeometry = new THREE.TorusGeometry(radius, 2, 8, 32);
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: zone.color,
        transparent: true,
        opacity: 0.3 - i * 0.1,
        side: THREE.DoubleSide,
      });

      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.position.copy(zone.center);
      ring.rotation.x = Math.PI / 2;
      ring.frustumCulled = false;
      this.zoneGroup.add(ring);
    }

    // Add cosmic particles
    this.createCosmicParticles(scene, zone);
  }

  createCosmicParticles(scene, zone) {
    const particleCount = 500;
    const particles = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      // Random position within the dark zone
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r =
        zone.innerRadius +
        Math.random() * (zone.outerRadius - zone.innerRadius);

      positions[i * 3] = zone.center.x + r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] =
        zone.center.y + (Math.random() - 0.5) * zone.height;
      positions[i * 3 + 2] =
        zone.center.z + r * Math.sin(phi) * Math.sin(theta);

      // Dark colors with occasional bright flashes
      const brightness = Math.random() > 0.95 ? 1.0 : 0.1;
      colors[i * 3] = brightness; // R
      colors[i * 3 + 1] = brightness * 0.2; // G
      colors[i * 3 + 2] = brightness * 0.8; // B
    }

    particles.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    particles.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
      size: 2,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const particleSystem = new THREE.Points(particles, particleMaterial);
    this.zoneGroup.add(particleSystem);
  }

  createQuantumDisks(scene) {
    const zone = this.zones.quantumDisk;

    // Create cyclonic particle formations for top and bottom
    this.createCyclonicFormation(scene, zone.topCenter, zone.radius, 'top');
    this.createCyclonicFormation(scene, zone.bottomCenter, zone.radius, 'bottom');
  }

  createCyclonicFormation(scene, center, radius, type) {
    // Create multiple orbital rings with particles
    const numRings = 5;
    const particlesPerRing = 50;

    for (let ring = 0; ring < numRings; ring++) {
      const ringRadius = (radius / numRings) * (ring + 1);
      const ringGeometry = new THREE.BufferGeometry();
      const positions = new Float32Array(particlesPerRing * 3);
      const colors = new Float32Array(particlesPerRing * 3);

      for (let i = 0; i < particlesPerRing; i++) {
        const angle = (i / particlesPerRing) * Math.PI * 2;
        const x = center.x + Math.cos(angle) * ringRadius;
        const z = center.z + Math.sin(angle) * ringRadius;
        const y = center.y + (Math.random() - 0.5) * 10; // Slight vertical variation

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        // Color varies by ring - inner rings brighter
        const intensity = 1 - (ring / numRings) * 0.7;
        colors[i * 3] = intensity; // R
        colors[i * 3 + 1] = intensity * 0.2; // G
        colors[i * 3 + 2] = intensity; // B
      }

      ringGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      ringGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const ringMaterial = new THREE.PointsMaterial({
        size: 3,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
      });

      const particleRing = new THREE.Points(ringGeometry, ringMaterial);
      particleRing.frustumCulled = false;
      this.zoneGroup.add(particleRing);
    }

    // Add central bright core
    const coreGeometry = new THREE.SphereGeometry(5, 16, 8);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0xff00ff,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    core.position.copy(center);
    core.frustumCulled = false;
    this.zoneGroup.add(core);
  }

  createMantleZone(scene) {
    const zone = this.zones.mantleZone;

    // Outer sphere wireframe
    const sphereGeometry = new THREE.SphereGeometry(zone.radius, 32, 16);
    const sphereMaterial = new THREE.MeshBasicMaterial({
      color: zone.color,
      transparent: true,
      opacity: 0.1,
      wireframe: true,
    });

    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.copy(zone.center);
    sphere.frustumCulled = false;
    this.zoneGroup.add(sphere);
  }

  createCentralCylinder(scene) {
    const zone = this.zones.centralCylinder;

    // Central connecting cylinder
    const cylinderGeometry = new THREE.CylinderGeometry(
      zone.radius,
      zone.radius,
      zone.height,
      32,
      1,
      true
    );
    const cylinderMaterial = new THREE.MeshPhysicalMaterial({
      color: zone.color,
      emissive: zone.color,
      emissiveIntensity: 0.2,
      transparent: true,
      opacity: 0.4,
      metalness: 0.8,
      roughness: 0.2,
      side: THREE.DoubleSide,
    });

    const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);
    cylinder.position.copy(zone.center);
    cylinder.frustumCulled = false;
    this.zoneGroup.add(cylinder);

    // Add energy flow particles inside cylinder
    this.createEnergyFlow(scene, zone);
  }

  createEnergyFlow(scene, zone) {
    const particleCount = 100;
    const particles = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      const radius = Math.random() * zone.radius * 0.8;
      const height = (Math.random() - 0.5) * zone.height;

      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = height;
      positions[i * 3 + 2] = Math.sin(angle) * radius;
    }

    particles.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const particleMaterial = new THREE.PointsMaterial({
      color: 0xffff00,
      size: 1,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
    });

    const particleSystem = new THREE.Points(particles, particleMaterial);
    this.zoneGroup.add(particleSystem);
  }

  // Position cells in their assigned zones
  positionCell(cellData, index, population) {
    const zone = this.determineZone(cellData);

    switch (zone) {
      case "memoryEliteZone":
        return this.memoryEliteZonePosition(cellData, index);
      case "lightZone":
        return this.lightZonePosition(cellData, index, population);
      case "darkZone":
        return this.darkZonePosition(cellData, index);
      case "quantumDisk":
        return this.quantumDiskPosition(cellData, index);
      case "centralCylinder":
        return this.centralCylinderPosition(cellData, index);
      case "mantleZone":
      default:
        return this.mantleZonePosition(cellData, index);
    }
  }
}

export default TransposableNeuralCell;
