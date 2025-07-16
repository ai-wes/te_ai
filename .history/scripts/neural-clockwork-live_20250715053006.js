class TransposableNeuralCell {
  constructor() {
    // Define the new architectural zones based on the diagram
    this.zones = {
      memoryEliteZone: {
        // First immediate sphere surrounding central cylinder
        center: new THREE.Vector3(0, 0, 0),
        radius: 150,
        wireframe: true,
        wireframeColor: 0x888888, // Grey wireframe
        wireframeOpacity: 0.15, // More transparent wireframe
        color: null, // No central color
        opacity: 0.1,
        description: "Memory Elite Zone - grey wireframe sphere, no fill",
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
        color: 0x000033, // Dark cosmic blue
        flashColor: 0xff00ff, // Bright pink flashes
        description: "Mutation zone - dark cosmic abyss",
      },
      quantumDisk: {
        // Top and bottom quantum disks
        topCenter: new THREE.Vector3(0, 130, 0),
        bottomCenter: new THREE.Vector3(0, -130, 0),
        radius: 65,
        thickness: 7,
        color: 0xff00ff, // Magenta/Purple
        description: "4D Quantum disk with central connection",
      },
      mantleZone: {
        // Outer sphere surrounding memory elite zone
        center: new THREE.Vector3(0, 0, 0),
        radius: 200, // Outside of memory elite zone (50)
        color: 0x333366, // Dark blue-gray
        wireframe: true,
        wireframeColor: 0x888888, // Grey wireframe
        opacity: 0.8, // 90% translucent
        transparent: true,
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

    // Wireframe sphere surrounding central cylinder
    const sphereGeometry = new THREE.SphereGeometry(zone.radius, 16, 8);
    const sphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x2a5555, // Darker version of cyan
      transparent: true,
      opacity: 0.1,
      wireframe: true,
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

    // Create outer sphere boundary
    const sphereGeometry = new THREE.SphereGeometry(zone.outerRadius, 16, 8);
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

    // Create central dark storm cloud
    this.createStormCloud(scene, zone);

    // Add cosmic particles
    this.createCosmicParticles(scene, zone);
  }

  createStormCloud(scene, zone) {
    // Create dense particle cloud at center using small spheres
    const stormCount = 250;
    const stormGroup = new THREE.Group();

    for (let i = 0; i < stormCount; i++) {
      // Position particles in a dense cloud at center
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = Math.random() * 30; // Small radius for storm core

      const x = zone.center.x + r * Math.sin(phi) * Math.cos(theta);
      const y = zone.center.y + (Math.random() - 0.5) * 40;
      const z = zone.center.z + r * Math.sin(phi) * Math.sin(theta);

      // Create small sphere for each particle
      const particleGeometry = new THREE.SphereGeometry(1, 1, 1);
      let particleColor;

      // Dark storm colors with pink and white flashes
      const flashType = Math.random();
      if (flashType > 0.9) {
        particleColor = 0xffffff; // White flash
      } else if (flashType > 0.85) {
        particleColor = 0xff3399; // Pink flash
      } else {
        particleColor = 0x0d0d26; // Dark storm color
      }

      const particleMaterial = new THREE.MeshBasicMaterial({
        color: particleColor,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
      });

      const particle = new THREE.Mesh(particleGeometry, particleMaterial);
      particle.position.set(x, y, z);
      particle.frustumCulled = false;
      stormGroup.add(particle);
    }

    this.zoneGroup.add(stormGroup);
  }

  createCosmicParticles(scene, zone) {
    const particleCount = 300;
    const orbitalGroup = new THREE.Group();

    for (let i = 0; i < particleCount; i++) {
      // Random position within the dark zone sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r =
        zone.innerRadius +
        Math.random() * (zone.outerRadius - zone.innerRadius);

      const x = zone.center.x + r * Math.sin(phi) * Math.cos(theta);
      const y = zone.center.y + (Math.random() - 0.5) * zone.height;
      const z = zone.center.z + r * Math.sin(phi) * Math.sin(theta);

      // Create small sphere for orbital particles
      const particleGeometry = new THREE.SphereGeometry(0.001, 0.01, 0.01);
      const particleMaterial = new THREE.MeshBasicMaterial({
        color: 0x332266, // Dark purple
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending,
      });

      const particle = new THREE.Mesh(particleGeometry, particleMaterial);
      particle.position.set(x, y, z);
      particle.frustumCulled = false;
      orbitalGroup.add(particle);
    }

    this.zoneGroup.add(orbitalGroup);
  }

  createQuantumDisks(scene) {
    const zone = this.zones.quantumDisk;

    // Create cyclonic particle formations for top and bottom
    this.createCyclonicFormation(scene, zone.topCenter, zone.radius, "top");
    this.createCyclonicFormation(
      scene,
      zone.bottomCenter,
      zone.radius,
      "bottom"
    );
  }

  createCyclonicFormation(scene, center, radius, type) {
    // Create multiple orbital rings with particles
    const numRings = 6;
    const particlesPerRing = 150;

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

      ringGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3)
      );
      ringGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

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
    const sphereGeometry = new THREE.SphereGeometry(zone.radius, 12, 6);
    const sphereMaterial = new THREE.MeshBasicMaterial({
      color: zone.color,
      transparent: true,
      opacity: 0.05,
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

class QuantumDreamPhaseVisualizer {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.isActive = false;

    // Dream state components
    this.dreamRealms = [];
    this.realityFragments = new THREE.Group();
    this.dreamParticles = null;
    this.quantumFog = null;
    this.realityMesh = null;

    // Store original scene state
    this.originalSceneState = {
      fog: scene.fog,
      background: scene.background,
    };

    this.setupDreamEnvironment();
  }

  setupDreamEnvironment() {
    // Quantum dream fog with shifting colors
    this.quantumFog = new THREE.FogExp2(0x000033, 0.002);

    // Reality mesh - the fabric of spacetime that will fracture
    const realityGeometry = new THREE.PlaneGeometry(1000, 1000, 100, 100);
    const realityMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        dreamDepth: { value: 0 },
        realityFragmentation: { value: 0 },
        quantumNoise: { value: 0 },
      },
      vertexShader: `
        uniform float time;
        uniform float dreamDepth;
        uniform float realityFragmentation;
        
        varying vec2 vUv;
        varying float vDistortion;
        varying vec3 vPosition;
        
        // Quantum noise function
        float qnoise(vec3 p) {
          return sin(p.x * 0.1) * cos(p.y * 0.1) * sin(p.z * 0.1 + time);
        }
        
        void main() {
          vUv = uv;
          vPosition = position;
          
          vec3 pos = position;
          
          // Reality distortion based on dream depth
          float distortion = qnoise(position * 0.05) * dreamDepth * 50.0;
          
          // Fragmentation effect
          float fragment = sin(position.x * 0.1 + time) * 
                         cos(position.y * 0.1 - time * 0.7) * 
                         realityFragmentation;
          
          pos.z += distortion + fragment * 20.0;
          
          // Ripple effect from center
          float dist = length(position.xy);
          pos.z += sin(dist * 0.05 - time * 2.0) * dreamDepth * 10.0;
          
          vDistortion = distortion;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float dreamDepth;
        uniform float quantumNoise;
        
        varying vec2 vUv;
        varying float vDistortion;
        varying vec3 vPosition;
        
        vec3 dreamColors(float phase) {
          vec3 color1 = vec3(0.1, 0.0, 0.3);  // Deep purple
          vec3 color2 = vec3(0.0, 0.2, 0.4);  // Quantum blue
          vec3 color3 = vec3(0.3, 0.0, 0.3);  // Magenta
          
          float t = phase * 3.0;
          
          if (t < 1.0) {
            return mix(color1, color2, t);
          } else if (t < 2.0) {
            return mix(color2, color3, t - 1.0);
          } else {
            return mix(color3, color1, t - 2.0);
          }
        }
        
        void main() {
          // Grid pattern that breaks down in dreams
          float grid = step(0.98, max(
            sin(vPosition.x * 0.5 + vDistortion * 0.1),
            sin(vPosition.y * 0.5 + vDistortion * 0.1)
          ));
          
          // Dream color phase
          float phase = sin(time * 0.5 + vDistortion * 0.1) * 0.5 + 0.5;
          vec3 color = dreamColors(phase);
          
          // Quantum interference patterns
          float interference = sin(vPosition.x * 0.2 + time) * 
                             cos(vPosition.y * 0.2 - time * 0.7);
          
          color += vec3(interference * quantumNoise * 0.2);
          
          // Fade at edges
          float edgeFade = 1.0 - smoothstep(200.0, 500.0, length(vPosition.xy));
          
          float alpha = (grid * 0.5 + 0.5) * dreamDepth * edgeFade;
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.realityMesh = new THREE.Mesh(realityGeometry, realityMaterial);
    this.realityMesh.rotation.x = -Math.PI / 2;
    this.realityMesh.position.y = -100;
  }

  createDreamRealm(realmIndex, totalRealms) {
    const realm = new THREE.Group();
    realm.userData = {
      index: realmIndex,
      phase: (realmIndex / totalRealms) * Math.PI * 2,
      rotationSpeed: 0.001 + Math.random() * 0.002,
      cells: new Map(),
      antigens: [],
    };

    // Create realm boundary - a translucent sphere
    const boundaryGeometry = new THREE.IcosahedronGeometry(150, 3);
    const boundaryMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        realmPhase: { value: realm.userData.phase },
        realmIndex: { value: realmIndex },
      },
      vertexShader: `
        uniform float time;
        uniform float realmPhase;
        
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          vNormal = normal;
          vPosition = position;
          
          // Breathing effect
          vec3 pos = position * (1.0 + sin(time + realmPhase) * 0.05);
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float realmPhase;
        uniform float realmIndex;
        
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          // Each realm has a unique color
          vec3 color = vec3(
            sin(realmIndex * 0.7) * 0.5 + 0.5,
            cos(realmIndex * 1.3) * 0.5 + 0.5,
            sin(realmIndex * 2.1 + 1.57) * 0.5 + 0.5
          );
          
          // Edge glow
          vec3 viewDirection = normalize(cameraPosition - vPosition);
          float rim = 1.0 - abs(dot(viewDirection, vNormal));
          
          // Quantum fluctuations
          float fluctuation = sin(vPosition.x * 0.1 + time + realmPhase) *
                            cos(vPosition.y * 0.1 - time) *
                            sin(vPosition.z * 0.1 + time);
          
          color += vec3(fluctuation * 0.2);
          
          float alpha = rim * 0.3 + 0.1;
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const boundary = new THREE.Mesh(boundaryGeometry, boundaryMaterial);
    realm.add(boundary);
    realm.userData.boundary = boundary;

    // Position realms in a spiral
    const angle = realm.userData.phase;
    const radius = 200 + realmIndex * 50;
    const height = Math.sin(realmIndex * 0.5) * 100;

    realm.position.set(
      Math.cos(angle) * radius,
      height,
      Math.sin(angle) * radius
    );

    // Add dream particles specific to this realm
    this.addRealmParticles(realm);

    return realm;
  }

  addRealmParticles(realm) {
    const particleCount = 500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    const color = new THREE.Color();

    for (let i = 0; i < particleCount; i++) {
      // Particles within realm boundary
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = Math.random() * 140;

      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);

      // Dream-like colors
      color.setHSL(
        realm.userData.phase / (Math.PI * 2),
        0.7,
        0.3 + Math.random() * 0.4
      );

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;

      sizes[i] = Math.random() * 2 + 0.5;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        
        void main() {
          vColor = color;
          
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform float time;
        varying vec3 vColor;
        
        void main() {
          vec2 center = gl_PointCoord - 0.5;
          float dist = length(center);
          
          if (dist > 0.5) discard;
          
          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
          alpha *= sin(time * 3.0) * 0.3 + 0.7;
          
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
      transparent: true,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const particles = new THREE.Points(geometry, material);
    realm.add(particles);
    realm.userData.particles = particles;
  }

  createDreamAntigen(realm, index) {
    const antigen = new THREE.Group();

    // Dream antigens are more abstract and shifting
    const geometry = new THREE.IcosahedronGeometry(3, 2);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        dreamPhase: { value: realm.userData.phase },
        morphFactor: { value: Math.random() },
      },
      vertexShader: `
        uniform float time;
        uniform float morphFactor;
        
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          vNormal = normal;
          vPosition = position;
          
          // Morphing dream geometry
          vec3 pos = position;
          float morph = sin(time * 2.0 + morphFactor * 6.28) * 0.5 + 0.5;
          
          pos += normal * sin(position.x * 3.0 + time) * morph * 2.0;
          pos += normal * cos(position.y * 3.0 - time) * morph * 2.0;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float dreamPhase;
        
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          // Iridescent dream colors
          vec3 color = vec3(
            sin(vPosition.x * 0.5 + time + dreamPhase) * 0.5 + 0.5,
            cos(vPosition.y * 0.5 - time + dreamPhase) * 0.5 + 0.5,
            sin(vPosition.z * 0.5 + time * 0.7 + dreamPhase) * 0.5 + 0.5
          );
          
          // Holographic effect
          vec3 viewDirection = normalize(cameraPosition - vPosition);
          float fresnel = pow(1.0 - dot(viewDirection, vNormal), 2.0);
          
          color = mix(color, vec3(1.0), fresnel * 0.5);
          
          gl_FragColor = vec4(color, 0.8);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    antigen.add(mesh);

    // Random position within realm
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * 100;
    const height = (Math.random() - 0.5) * 100;

    antigen.position.set(Math.cos(angle) * r, height, Math.sin(angle) * r);

    antigen.userData = {
      isDreamAntigen: true,
      morphSpeed: 0.5 + Math.random() * 1.5,
      floatSpeed: 0.5 + Math.random(),
    };

    return antigen;
  }

  createDreamCell(originalCell, realm) {
    // Create ethereal version of cell
    const dreamCell = originalCell.clone();
    dreamCell.userData = {
      ...originalCell.userData,
      isDreamVersion: true,
      originalId: originalCell.userData.cell_id,
    };

    // Make it ghostly
    dreamCell.traverse((obj) => {
      if (obj.material) {
        obj.material = obj.material.clone();
        obj.material.transparent = true;
        obj.material.opacity = 0.3;
        obj.material.emissive = new THREE.Color(
          0.5 + realm.userData.index * 0.1,
          0.3,
          0.8
        );
        obj.material.emissiveIntensity = 0.5;
      }
    });

    return dreamCell;
  }

  enterDreamPhase(numRealms = 5) {
    this.isActive = true;

    // Save current scene state
    this.originalSceneState.fog = this.scene.fog;
    this.originalSceneState.background = this.scene.background;

    // Transform to dream environment
    this.scene.fog = this.quantumFog;
    this.scene.background = new THREE.Color(0x000011);

    // Add reality mesh
    this.scene.add(this.realityMesh);

    // Create multiple dream realms
    for (let i = 0; i < numRealms; i++) {
      const realm = this.createDreamRealm(i, numRealms);
      this.dreamRealms.push(realm);
      this.scene.add(realm);
    }

    // Animate entrance
    this.animateDreamEntry();
  }

  animateDreamEntry() {
    const duration = 3000;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease in function
      const easeProgress = 1 - Math.pow(1 - progress, 3);

      // Update reality mesh fragmentation
      if (this.realityMesh.material.uniforms) {
        this.realityMesh.material.uniforms.dreamDepth.value = easeProgress;
        this.realityMesh.material.uniforms.realityFragmentation.value =
          easeProgress;
        this.realityMesh.material.uniforms.quantumNoise.value = easeProgress;
      }

      // Fade regular cells
      this.scene.traverse((obj) => {
        if (obj.userData.cell_id && !obj.userData.isGhost) {
          if (obj.userData.hub) {
            obj.userData.hub.material.opacity = 1 - easeProgress * 0.7;
            obj.userData.hub.material.transparent = true;
          }
        }
      });

      // Scale in dream realms
      this.dreamRealms.forEach((realm, i) => {
        const delay = i * 0.1;
        const realmProgress = Math.max(0, (progress - delay) / (1 - delay));
        realm.scale.setScalar(realmProgress);
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        this.startDreamAnimation();
      }
    };

    animate();
  }

  startDreamAnimation() {
    // Populate realms with dream antigens and responses
    this.dreamRealms.forEach((realm, index) => {
      // Create dream antigens
      for (let i = 0; i < 10; i++) {
        const dreamAntigen = this.createDreamAntigen(realm, i);
        realm.userData.antigens.push(dreamAntigen);
        realm.add(dreamAntigen);
      }
    });
  }

  update(time) {
    if (!this.isActive) return;

    // Update reality mesh
    if (this.realityMesh.material.uniforms) {
      this.realityMesh.material.uniforms.time.value = time;
    }

    // Update dream realms
    this.dreamRealms.forEach((realm) => {
      // Rotate realm
      realm.rotation.y += realm.userData.rotationSpeed;

      // Update boundary
      if (realm.userData.boundary.material.uniforms) {
        realm.userData.boundary.material.uniforms.time.value = time;
      }

      // Update particles
      if (realm.userData.particles.material.uniforms) {
        realm.userData.particles.material.uniforms.time.value = time;
      }

      // Animate dream antigens
      realm.userData.antigens.forEach((antigen) => {
        // Floating motion
        antigen.position.y +=
          Math.sin(time * antigen.userData.floatSpeed) * 0.1;
        antigen.rotation.x += 0.01;
        antigen.rotation.y += 0.005;

        // Update shader
        if (antigen.children[0] && antigen.children[0].material.uniforms) {
          antigen.children[0].material.uniforms.time.value = time;
        }
      });
    });

    // Create inter-realm connections showing learning transfer
    this.updateInterRealmConnections(time);
  }

  updateInterRealmConnections(time) {
    // Periodically create energy transfers between realms
    if (Math.random() < 0.02) {
      const realm1 =
        this.dreamRealms[Math.floor(Math.random() * this.dreamRealms.length)];
      const realm2 =
        this.dreamRealms[Math.floor(Math.random() * this.dreamRealms.length)];

      if (realm1 !== realm2) {
        this.createLearningTransfer(realm1, realm2, time);
      }
    }
  }

  createLearningTransfer(realm1, realm2, time) {
    const transfer = new THREE.Group();

    // Create energy bolt
    const points = [];
    const segments = 20;

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const pos = new THREE.Vector3().lerpVectors(
        realm1.position,
        realm2.position,
        t
      );

      // Add lightning-like randomness
      if (i > 0 && i < segments) {
        pos.x += (Math.random() - 0.5) * 20;
        pos.y += (Math.random() - 0.5) * 20;
        pos.z += (Math.random() - 0.5) * 20;
      }

      points.push(pos);
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const bolt = new THREE.Line(geometry, material);
    transfer.add(bolt);

    this.scene.add(transfer);

    // Animate and remove
    const duration = 1000;
    const startTime = Date.now();

    const animateBolt = () => {
      const elapsed = Date.now() - startTime;
      const progress = elapsed / duration;

      material.opacity = 0.8 * (1 - progress);

      if (progress < 1) {
        requestAnimationFrame(animateBolt);
      } else {
        this.scene.remove(transfer);
      }
    };

    animateBolt();
  }

  exitDreamPhase() {
    const duration = 2000;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Reverse the entrance animation
      const easeProgress = Math.pow(progress, 3);

      // Update reality mesh
      if (this.realityMesh.material.uniforms) {
        this.realityMesh.material.uniforms.dreamDepth.value = 1 - easeProgress;
        this.realityMesh.material.uniforms.realityFragmentation.value =
          1 - easeProgress;
      }

      // Fade dream realms
      this.dreamRealms.forEach((realm) => {
        realm.scale.setScalar(1 - easeProgress);
      });

      // Restore regular cells
      this.scene.traverse((obj) => {
        if (
          obj.userData.cell_id &&
          !obj.userData.isGhost &&
          !obj.userData.isDreamVersion
        ) {
          if (obj.userData.hub) {
            obj.userData.hub.material.opacity = 0.3 + easeProgress * 0.7;
          }
        }
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        this.cleanup();
      }
    };

    animate();
  }

  cleanup() {
    // Remove dream elements
    this.dreamRealms.forEach((realm) => {
      this.scene.remove(realm);
    });
    this.dreamRealms = [];

    this.scene.remove(this.realityMesh);

    // Restore original scene
    this.scene.fog = this.originalSceneState.fog;
    this.scene.background = this.originalSceneState.background;

    this.isActive = false;
  }
}

class QuantumEntanglementVisualizer {
  constructor(scene) {
    this.scene = scene;
    this.entangledPairs = new Map(); // Maps cell_id to its ghost twin
    this.entanglementBeams = new Map();
    this.quantumField = this.createQuantumField();
  }

  createQuantumField() {
    // Ambient quantum field effect
    const fieldGeometry = new THREE.SphereGeometry(200, 32, 16);
    const fieldMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        opacity: { value: 0.02 },
      },
      vertexShader: `
        varying vec3 vPosition;
        varying vec3 vNormal;
        
        void main() {
          vPosition = position;
          vNormal = normal;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float opacity;
        varying vec3 vPosition;
        varying vec3 vNormal;
        
        void main() {
          // Quantum field fluctuations
          float noise = sin(vPosition.x * 0.1 + time) * 
                       cos(vPosition.y * 0.1 - time * 0.7) * 
                       sin(vPosition.z * 0.1 + time * 1.3);
          
          vec3 color = vec3(0.8, 0.0, 1.0); // Purple quantum field
          float alpha = opacity * (0.5 + 0.5 * noise);
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });

    const field = new THREE.Mesh(fieldGeometry, fieldMaterial);
    this.scene.add(field);
    return field;
  }

  createQuantumGhost(originalCell, cellData) {
    // Check if already has a ghost
    if (this.entangledPairs.has(originalCell.userData.cell_id)) {
      return this.entangledPairs.get(originalCell.userData.cell_id);
    }

    // Create ghost twin
    const ghost = new THREE.Group();
    ghost.userData = {
      ...cellData,
      isGhost: true,
      entangledWith: originalCell.userData.cell_id,
    };

    // Clone the cell geometry but with quantum ghost materials
    const originalHub = originalCell.userData.hub;
    const ghostHub = originalHub.geometry.clone();

    // Create special quantum ghost material
    const ghostMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        baseColor: { value: new THREE.Color(0xff00ff) },
        phaseOffset: { value: Math.random() * Math.PI * 2 },
        entanglementStrength: { value: 1.0 },
      },
      vertexShader: `
        varying vec3 vPosition;
        varying vec3 vNormal;
        
        void main() {
          vPosition = position;
          vNormal = normalize(normalMatrix * normal);
          
          // Quantum fluctuation
          vec3 pos = position;
          float fluctuation = sin(time * 3.0 + position.y * 0.5) * 0.1;
          pos += normal * fluctuation;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 baseColor;
        uniform float phaseOffset;
        uniform float entanglementStrength;
        
        varying vec3 vPosition;
        varying vec3 vNormal;
        
        void main() {
          // Quantum interference pattern
          float interference = sin(vPosition.x * 10.0 + time + phaseOffset) *
                             cos(vPosition.y * 10.0 - time * 0.7) *
                             sin(vPosition.z * 10.0 + time * 1.3);
          
          // Edge glow effect
          vec3 viewDirection = normalize(cameraPosition - vPosition);
          float edgeFactor = pow(1.0 - abs(dot(viewDirection, vNormal)), 2.0);
          
          // Quantum phase visualization
          vec3 color = baseColor;
          color = mix(color, vec3(0.0, 1.0, 1.0), interference * 0.5 + 0.5);
          
          // Transparency with quantum fluctuation
          float alpha = 0.3 + edgeFactor * 0.4;
          alpha *= (0.7 + sin(time * 2.0 + phaseOffset) * 0.3) * entanglementStrength;
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const ghostMesh = new THREE.Mesh(ghostHub, ghostMaterial);
    ghost.add(ghostMesh);
    ghost.userData.hub = ghostMesh;

    // Add quantum phase indicator
    this.addQuantumPhaseRings(ghost);

    // Position ghost at opposite location
    this.positionQuantumGhost(ghost, originalCell);

    // Create entanglement visualization
    this.createEntanglementBeam(originalCell, ghost);

    // Store the pair
    this.entangledPairs.set(originalCell.userData.cell_id, ghost);
    this.entangledPairs.set(ghost.userData.cell_id, originalCell);

    return ghost;
  }

  positionQuantumGhost(ghost, originalCell) {
    // Calculate opposite position through quantum inversion
    const originalPos = originalCell.position.clone();

    // Find the zone center
    let zoneCenter = new THREE.Vector3(0, -60, 0); // Quantum layer center

    // Calculate vector from zone center to cell
    const offset = originalPos.clone().sub(zoneCenter);

    // Invert through center (point reflection)
    const ghostOffset = offset.clone().multiplyScalar(-1);

    // Add some quantum uncertainty
    const uncertainty = new THREE.Vector3(
      (Math.random() - 0.5) * 5,
      (Math.random() - 0.5) * 5,
      (Math.random() - 0.5) * 5
    );

    ghost.position.copy(zoneCenter).add(ghostOffset).add(uncertainty);

    // Opposite rotation phase
    ghost.rotation.y = originalCell.rotation.y + Math.PI;
    ghost.rotation.x = -originalCell.rotation.x;
    ghost.rotation.z = -originalCell.rotation.z;
  }

  addQuantumPhaseRings(ghost) {
    // Rotating rings showing quantum phase
    const ringGroup = new THREE.Group();

    for (let i = 0; i < 3; i++) {
      const ringGeometry = new THREE.TorusGeometry(
        8 + i * 3, // radius
        0.2, // tube
        8, // radialSegments
        32 // tubularSegments
      );

      const ringMaterial = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(0.8 + i * 0.1, 1, 0.5),
        transparent: true,
        opacity: 0.3 - i * 0.1,
        blending: THREE.AdditiveBlending,
      });

      const ring = new THREE.Mesh(ringGeometry, ringMaterial);

      // Different rotation axes for each ring
      ring.rotation.x = (i * Math.PI) / 3;
      ring.rotation.z = (i * Math.PI) / 4;

      ring.userData.rotationSpeed = {
        x: 0.01 * (i + 1),
        y: 0.02 * (i + 1),
        z: 0.005 * (i + 1),
      };

      ringGroup.add(ring);
    }

    ghost.add(ringGroup);
    ghost.userData.phaseRings = ringGroup;
  }

  createEntanglementBeam(cell1, cell2) {
    const beamId = `${cell1.userData.cell_id}_${cell2.userData.cell_id}`;

    // Create spline curve between cells
    const points = [];
    const startPos = cell1.position;
    const endPos = cell2.position;
    const midPoint = startPos.clone().add(endPos).multiplyScalar(0.5);

    // Add quantum uncertainty to path
    const control1 = midPoint
      .clone()
      .add(
        new THREE.Vector3(
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20
        )
      );

    const control2 = midPoint
      .clone()
      .add(
        new THREE.Vector3(
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20
        )
      );

    const curve = new THREE.CubicBezierCurve3(
      startPos,
      control1,
      control2,
      endPos
    );

    // Create beam geometry
    const tubeGeometry = new THREE.TubeGeometry(curve, 64, 0.5, 8, false);

    // Quantum entanglement shader
    const beamMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        startPoint: { value: startPos },
        endPoint: { value: endPos },
      },
      vertexShader: `
        uniform vec3 startPoint;
        uniform vec3 endPoint;
        varying float vProgress;
        varying vec3 vPosition;
        
        void main() {
          vPosition = (modelMatrix * vec4(position, 1.0)).xyz;
          
          // Calculate progress along beam
          float totalDist = distance(startPoint, endPoint);
          float currentDist = distance(startPoint, vPosition);
          vProgress = currentDist / totalDist;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        varying float vProgress;
        varying vec3 vPosition;
        
        void main() {
          // Quantum correlation wave
          float wave = sin(vProgress * 20.0 - time * 5.0) * 0.5 + 0.5;
          float pulse = sin(time * 3.0) * 0.5 + 0.5;
          
          // EPR correlation visualization
          vec3 color = mix(
            vec3(1.0, 0.0, 1.0),  // Magenta
            vec3(0.0, 1.0, 1.0),  // Cyan
            wave
          );
          
          float alpha = 0.3 * pulse + 0.2 * wave;
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    const beam = new THREE.Mesh(tubeGeometry, beamMaterial);
    beam.userData.curve = curve;

    this.scene.add(beam);
    this.entanglementBeams.set(beamId, beam);

    // Add quantum correlation particles
    this.addCorrelationParticles(curve);
  }

  addCorrelationParticles(curve) {
    const particleCount = 50;
    const particles = new THREE.Group();

    for (let i = 0; i < particleCount; i++) {
      const particleGeometry = new THREE.SphereGeometry(0.3, 8, 8);
      const particleMaterial = new THREE.MeshBasicMaterial({
        color: 0xff00ff,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending,
      });

      const particle = new THREE.Mesh(particleGeometry, particleMaterial);
      particle.userData.progress = i / particleCount;
      particle.userData.speed = 0.5 + Math.random() * 0.5;
      particle.userData.curve = curve;

      particles.add(particle);
    }

    this.scene.add(particles);
    particles.userData.isCorrelationParticles = true;
  }

  update(time) {
    // Update quantum field
    if (this.quantumField.material.uniforms) {
      this.quantumField.material.uniforms.time.value = time;
    }

    // Update all ghost materials
    this.entangledPairs.forEach((entity, id) => {
      if (entity.userData.isGhost && entity.userData.hub) {
        entity.userData.hub.material.uniforms.time.value = time;

        // Quantum phase oscillation
        const original = this.entangledPairs.get(entity.userData.entangledWith);
        if (original) {
          // Opposite phase
          entity.userData.hub.material.uniforms.phaseOffset.value =
            -original.rotation.y + Math.PI;
        }
      }

      // Update phase rings
      if (entity.userData.phaseRings) {
        entity.userData.phaseRings.children.forEach((ring) => {
          ring.rotation.x += ring.userData.rotationSpeed.x;
          ring.rotation.y += ring.userData.rotationSpeed.y;
          ring.rotation.z += ring.userData.rotationSpeed.z;
        });
      }
    });

    // Update entanglement beams
    this.entanglementBeams.forEach((beam) => {
      if (beam.material.uniforms) {
        beam.material.uniforms.time.value = time;
      }
    });

    // Update correlation particles
    this.scene.traverse((obj) => {
      if (obj.parent && obj.parent.userData.isCorrelationParticles) {
        const progress =
          (obj.userData.progress + time * obj.userData.speed * 0.1) % 1;
        obj.userData.progress = progress;

        const point = obj.userData.curve.getPoint(progress);
        obj.position.copy(point);

        // Pulse effect
        const scale = 0.5 + Math.sin(time * 5 + progress * Math.PI * 2) * 0.3;
        obj.scale.setScalar(scale);
      }
    });
  }
}

class GerminalCenterLayout {
  constructor() {
    // Define functional zones like a real germinal center - Better aligned
    this.zones = {
      darkZone: {
        center: new THREE.Vector3(-100, 0, 0),
        radius: 70,
        color: 0x4444aa, // Brighter blue-purple
        description: "High mutation zone - active transposition",
      },
      lightZone: {
        center: new THREE.Vector3(100, 0, 0),
        radius: 70,
        color: 0xaaaa44, // Brighter yellow-green
        description: "Selection zone - high fitness cells",
      },
      mantleZone: {
        center: new THREE.Vector3(0, 0, 0),
        radius: 200,
        color: 0x666666, // Medium gray
        description: "Transitional cells",
      },
      memoryZone: {
        center: new THREE.Vector3(0, 120, 0),
        radius: 60,
        color: 0x44aaaa, // Brighter cyan
        description: "Memory B-cells - stable high performers",
      },
      quantumLayer: {
        center: new THREE.Vector3(0, -120, 0),
        radius: 70,
        color: 0xaa44aa, // Brighter magenta
        description: "Quantum superposition space",
      },
    };

    // Network connections for HGT
    this.plasmidNetwork = new Map();
    this.lineageTree = new Map();
  }

  assignCellPosition(cellData, index, population) {
    const position = new THREE.Vector3();

    // Determine cell zone based on properties
    const zone = this.determineCellZone(cellData, population);

    // Position within zone based on additional properties
    switch (zone) {
      case "darkZone":
        // Arrange by mutation rate and stress
        position.copy(this.darkZonePosition(cellData, index));
        break;

      case "lightZone":
        // Arrange by fitness in concentric rings
        position.copy(this.lightZonePosition(cellData, index, population));
        break;

      case "memoryZone":
        // Stable arrangement for long-lived cells
        position.copy(this.memoryZonePosition(cellData, index));
        break;

      case "quantumLayer":
        // 4D projection arrangement for quantum cells
        position.copy(this.quantumLayerPosition(cellData, index));
        break;

      default:
        // Mantle zone - transitional positions
        position.copy(this.mantleZonePosition(cellData, index));
    }

    return position;
  }

  determineCellZone(cellData, population) {
    const fitness = cellData.fitness || 0.5;
    const generation = cellData.generation || 0;
    const hasQuantumGenes = cellData.genes?.some((g) => g.is_quantum) || false;
    const mutationRate = this.calculateMutationRate(cellData);

    // High fitness cells that have survived many generations -> memory
    if (fitness > 0.8 && generation > 20) {
      return "memoryZone";
    }

    // Quantum cells exist in their own phase space
    if (hasQuantumGenes && Math.random() < 0.7) {
      return "quantumLayer";
    }

    // High mutation rate -> dark zone
    if (mutationRate > 0.5 || (cellData.stress_level || 0) > 0.6) {
      return "darkZone";
    }

    // High fitness -> light zone for selection
    if (fitness > 0.65) {
      return "lightZone";
    }

    // Everyone else in mantle
    return "mantleZone";
  }

  calculateMutationRate(cellData) {
    // Calculate based on gene dynamics
    if (!cellData.genes) return 0.3;

    const transposableGenes = cellData.genes.filter((g) => g.is_active).length;
    const totalGenes = cellData.genes.length;
    const stressLevel = cellData.stress_level || 0;

    return Math.min(
      1,
      (transposableGenes / totalGenes) * 0.5 + stressLevel * 0.5
    );
  }

  darkZonePosition(cellData, index) {
    // Spherical helix arrangement based on mutation activity
    const zone = this.zones.darkZone;

    // Create spherical helix pattern
    const mutationIntensity = this.calculateMutationRate(cellData);

    // Spherical helix parameters with stable positioning
    const totalCells = 50; // Expected number of cells in this zone
    const turns = 3; // Number of helix turns
    const t = (index % totalCells) / totalCells; // Normalize to [0,1] and prevent overflow

    // Use stable angles that don't drift
    const theta = t * turns * Math.PI * 2; // Angle around helix
    const phi = Math.PI * (0.1 + t * 0.8); // Keep away from poles for better distribution

    // Radius varies with mutation intensity but stays within bounds
    const baseRadius = zone.radius * 0.5; // Base radius for helix
    const radiusModulation = Math.sin(theta * 2) * 3 * mutationIntensity; // Reduced modulation
    const r = baseRadius + radiusModulation;

    // Spherical coordinates - properly centered without scaling
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.cos(phi);
    const z = r * Math.sin(phi) * Math.sin(theta);

    return new THREE.Vector3(
      zone.center.x + x, // No scaling - keep centered
      zone.center.y + y,
      zone.center.z + z // No scaling - keep centered
    );
  }

  lightZonePosition(cellData, index, population) {
    // Spherical distribution organized by fitness
    const zone = this.zones.lightZone;
    const fitness = cellData.fitness || 0.5;

    // Sort by fitness rank
    const fitnessRank = cellData.fitnessRank || index;
    const normalizedRank = fitnessRank / population.length;

    // Spiral sphere distribution - fittest cells at center
    const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // Golden angle
    const theta = goldenAngle * index;

    // Vertical distribution based on fitness
    const y = 1 - 2 * normalizedRank; // Maps to [-1, 1]
    const radiusAtY = Math.sqrt(1 - y * y); // Sphere equation

    // Scale by zone radius and fitness
    const radiusScale = zone.radius * (0.3 + normalizedRank * 0.7);

    const x = Math.cos(theta) * radiusAtY * radiusScale;
    const z = Math.sin(theta) * radiusAtY * radiusScale;

    return new THREE.Vector3(
      zone.center.x + x,
      zone.center.y + y * radiusScale * 0.8,
      zone.center.z + z
    );
  }

  memoryZonePosition(cellData, index) {
    // Stable crystalline lattice for memory cells
    const zone = this.zones.memoryZone;

    // 3D hexagonal close packing
    const layer = Math.floor(index / 19); // 19 cells per layer in hex packing
    const indexInLayer = index % 19;

    // Convert to hex coordinates
    const hexAngle = (indexInLayer / 19) * Math.PI * 2;
    const hexRadius = indexInLayer < 7 ? 8 : 16;

    return new THREE.Vector3(
      zone.center.x + Math.cos(hexAngle) * hexRadius,
      zone.center.y + layer * 5,
      zone.center.z + Math.sin(hexAngle) * hexRadius
    );
  }

  quantumLayerPosition(cellData, index) {
    // Quantum superposition sphere with phase oscillations
    const zone = this.zones.quantumLayer;

    // Quantum phase parameters
    const phase = index * 0.618033988749895; // Golden ratio
    const quantumPhase = Math.sin(phase * 4) * 0.5 + 0.5;

    // Spherical distribution with quantum fluctuations
    const theta = phase * Math.PI * 2;
    const phi = Math.acos(1 - (2 * (index % 50)) / 50);

    // Radius oscillates with quantum phase
    const baseRadius = zone.radius * 0.8;
    const radiusOscillation = Math.sin(phase * 8) * 10 * quantumPhase;
    const r = baseRadius + radiusOscillation;

    // Spherical to Cartesian with quantum wobble
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.cos(phi) + Math.sin(phase * 6) * 5;
    const z = r * Math.sin(phi) * Math.sin(theta);

    return new THREE.Vector3(
      zone.center.x + x,
      zone.center.y + y,
      zone.center.z + z
    );
  }

  mantleZonePosition(cellData, index) {
    // Spherical shell arrangement for transitional cells
    const zone = this.zones.mantleZone;

    // Fibonacci sphere distribution for even spacing
    const goldenRatio = (1 + Math.sqrt(5)) / 2;
    const angleIncrement = Math.PI * (3 - Math.sqrt(5)); // Golden angle

    // Normalize index for sphere distribution
    const t = index / 100; // Adjust divisor based on expected population
    const inclination = Math.acos(1 - 2 * t);
    const azimuth = angleIncrement * index;

    // Place cells in a shell between inner zones and outer boundary
    const innerRadius = zone.radius * 0.5; // Inner shell
    const outerRadius = zone.radius * 0.8; // Outer shell

    // Vary radius to create depth
    const radiusVariation = Math.sin(index * 0.1) * 0.1;
    const radius =
      innerRadius + (outerRadius - innerRadius) * (0.5 + radiusVariation);

    const x = radius * Math.sin(inclination) * Math.cos(azimuth);
    const y = radius * Math.cos(inclination);
    const z = radius * Math.sin(inclination) * Math.sin(azimuth);

    return new THREE.Vector3(
      zone.center.x + x,
      zone.center.y + y,
      zone.center.z + z
    );
  }

  createZoneVisualizations(scene) {
    // Create a group for all zone visualizations
    this.zoneGroup = new THREE.Group();
    this.zoneGroup.frustumCulled = false;

    // Sort zones by radius (largest first) so smaller zones render on top
    const sortedZones = Object.entries(this.zones).sort(
      (a, b) => b[1].radius - a[1].radius
    );

    // Add subtle zone indicators
    sortedZones.forEach(([name, zone]) => {
      // Zone boundary sphere with wireframe overlay - simplified geometry
      const boundaryGeometry = new THREE.SphereGeometry(zone.radius, 16, 8);

      // Solid translucent sphere
      const boundaryMaterial = new THREE.MeshPhysicalMaterial({
        color: zone.color,
        transparent: true,
        opacity: 0.1,
        metalness: 0.2,
        roughness: 0.8,
        transmission: 0.6,
        side: THREE.BackSide,
        depthWrite: false,
      });

      const boundary = new THREE.Mesh(boundaryGeometry, boundaryMaterial);
      boundary.position.copy(zone.center);
      boundary.frustumCulled = false; // Always render regardless of camera position
      boundary.renderOrder = -1; // Render before other objects
      this.zoneGroup.add(boundary); // Add to group instead of scene

      // Add wireframe overlay for better visibility using LineSegments
      const wireframeGeometry = new THREE.SphereGeometry(
        zone.radius * 1.01,
        12,
        6
      );
      const edges = new THREE.EdgesGeometry(wireframeGeometry);
      const wireframeMaterial = new THREE.LineBasicMaterial({
        color: zone.color,
        transparent: true,
        opacity: 0.15,
        depthWrite: false,
      });

      const wireframe = new THREE.LineSegments(edges, wireframeMaterial);
      wireframe.position.copy(zone.center);
      wireframe.frustumCulled = false; // Always render regardless of camera position
      this.zoneGroup.add(wireframe); // Add to group

      // Add zone ring at equator for extra definition
      const ringGeometry = new THREE.TorusGeometry(zone.radius, 1, 8, 32); // Reduced segments
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: zone.color,
        transparent: true,
        opacity: 0.3,
        blending: THREE.AdditiveBlending,
      });

      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.position.copy(zone.center);
      ring.rotation.x = Math.PI / 2;
      ring.frustumCulled = false; // Always render regardless of camera position
      this.zoneGroup.add(ring); // Add to group

      // Zone particles for atmosphere
      this.createZoneParticles(scene, zone);
    });

    // Add the entire zone group to the scene
    scene.add(this.zoneGroup);
  }

  createZoneParticles(scene, zone) {
    const particleCount = 200;
    const particles = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      // Random position within zone
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = Math.random() * zone.radius;

      positions[i * 3] = zone.center.x + r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] =
        zone.center.y + r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = zone.center.z + r * Math.cos(phi);
    }

    particles.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const particleMaterial = new THREE.PointsMaterial({
      color: zone.color,
      size: 0.5,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
    });

    const particleSystem = new THREE.Points(particles, particleMaterial);
    scene.add(particleSystem);
  }
}

class LiveNeuralClockwork {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a); // Dark background to match UI

    // Add fog for atmosphere and depth (disabled for unlimited view distance)
    // this.scene.fog = new THREE.Fog(0xf0f0f0, 200, 600);

    // Core data structures
    this.genes = new Map(); // gene_id -> gene module
    this.cells = new Map(); // cell_id -> cell mechanism
    this.connections = new Map(); // connection_id -> connection beam
    this.quantum_genes = new Map(); // quantum gene special effects

    // Visual components
    this.neural_architecture = null;
    this.population_mechanism = null;

    // Camera and controls
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Animation state
    this.time = 0;
    this.lastState = null;
    this.selectedObject = null;

    // Event handlers
    this.eventHandlers = new Map();

    // Gene type configurations
    this.gene_type_colors = {
      V: 0x0080ff, // Blue hexagonal
      D: 0x00ff80, // Green octagonal
      J: 0xffaa00, // Orange dodecagonal
      Q: 0xff00ff, // Purple quantum tesseract
      S: 0xffffff, // Pure white for stem cells
    };

    // Mode state
    this.mode = "live"; // 'live' or 'historical'
    this.historicalRunDir = null;
    this.pollingActive = true;
    this.paused = false;

    // Initialize
    this.setupPolling();
    this.setupScene();
    this.setupLights();
    this.initializeArchitecture();
    this.animate();

    // Create a test stem cell to verify it works (disabled when loading real data)
    // this.createTestStemCell();
  }

  setMode(mode, runDir = null) {
    this.mode = mode;
    this.historicalRunDir = runDir;

    if (mode === "live") {
      this.pollingActive = true;
      this.startPolling();
    } else if (mode === "historical") {
      this.pollingActive = false;
      // Stop current polling
      if (this.pollTimeout) {
        clearTimeout(this.pollTimeout);
        this.pollTimeout = null;
      }
      console.log("Switched to historical mode - polling stopped");
    }
  }

  loadHistoricalData(data) {
    // Load historical generation data directly
    console.log("Loading historical data:", data);
    console.log("Number of cells in data:", data.cells ? data.cells.length : 0);

    // Clear ALL existing cells from the scene
    this.cells.forEach((cell, cellId) => {
      this.population_mechanism.remove(cell);
      this.scene.remove(cell);
    });
    this.cells.clear();

    // Clear the population mechanism children
    while (this.population_mechanism.children.length > 0) {
      this.population_mechanism.remove(this.population_mechanism.children[0]);
    }

    // Load new state
    this.handleStateUpdate(data);

    // Force refresh all cell colors after a delay
    setTimeout(() => this.refreshCellColors(), 200);
  }

  refreshCellColors() {
    // Refresh colors for all existing cells
    this.cells.forEach((cell, cellId) => {
      const cellData = cell.userData;
      if (cellData && cellData.genes) {
        const specialization = this.determineCellSpecialization(cellData);
        const fitness = cellData.fitness || 0.5;
        const hue = this.getSpecializationHue(specialization);

        if (cell.userData.hub && cell.userData.hub.material) {
          const saturation = 0.3 + fitness * 0.7;
          const lightness = 0.3 + fitness * 0.2;

          cell.userData.hub.material.color.setHSL(hue, saturation, lightness);
          cell.userData.hub.material.emissive.setHSL(hue, 0.8, 0.3);
          cell.userData.hub.material.needsUpdate = true;

          console.log(
            `Refreshed cell ${cellId}: type=${specialization.primary}, hue=${hue}`
          );
        }
      }
    });
  }

  setupPolling() {
    this.pollInterval = 20000; // Poll every 20 seconds
    if (this.mode === "live") {
      this.startPolling();
    }
  }

  async startPolling() {
    if (!this.pollingActive) return;

    const poll = async () => {
      if (!this.pollingActive || this.mode !== "live") return;

      try {
        const response = await fetch("te_ai_state.json?t=" + Date.now());
        if (response.ok) {
          const state = await response.json();

          // Log the received data to the console
          console.log("Data received from server:", state);

          if (JSON.stringify(state) !== JSON.stringify(this.lastState)) {
            this.handleStateUpdate(state);
            this.lastState = state;
          }

          this.emit("connection_status", {
            connected: true,
            text: "Connected",
          });
        }
      } catch (error) {
        this.emit("connection_status", {
          connected: false,
          text: "Disconnected",
        });
      }

      if (this.pollingActive && this.mode === "live") {
        this.pollTimeout = setTimeout(poll, this.pollInterval);
      }
    };

    poll();
  }

  handleStateUpdate(state) {
    // Check for stress events
    const currentStress = state.stress_level || state.stress || 0;
    if (
      this.lastStressLevel !== undefined &&
      currentStress > 0.8 &&
      this.lastStressLevel < 0.8
    ) {
      // High stress event triggered!
      this.triggerStressEvent(currentStress);
    }
    this.lastStressLevel = currentStress;

    // Emit standard events with proper field mapping
    if (state.generation >= 0) {
      this.emit("generation_start", {
        generation: state.generation,
        population_size: state.population_size || state.population || 0,
        stress_level: state.stress_level || state.stress || 0,
      });

      this.emit("generation_complete", {
        generation: state.generation,
        metrics: {
          mean_fitness: state.mean_fitness || state.fitness || 0,
          diversity: state.diversity || 0,
        },
      });

      if (state.phase) {
        this.emit("phase_transition", { phase: state.phase });
      }
    }

    // Process architecture updates
    if (state.events) {
      state.events.forEach((event) => this.handleEvent(event));
    }

    // Update population structure
    this.updatePopulationStructure(state);

    // Refresh colors after update
    setTimeout(() => this.refreshCellColors(), 100);
  }

  updatePopulationStructure(state) {
    if (!this.layout) {
      this.layout = new GerminalCenterLayout();
      this.layout.createZoneVisualizations(this.scene);
    }

    if (state.cells && state.cells.length > 0) {
      console.log(`=== Loading ${state.cells.length} cells from data ===`);

      // First, calculate fitness ranks for the entire population
      const sortedByFitness = [...state.cells].sort(
        (a, b) => (b.fitness || 0) - (a.fitness || 0)
      );

      state.cells.forEach((cellData, index) => {
        // Debug first cell structure
        if (index === 0) {
          console.log("First cell data structure:", cellData);
          if (cellData.genes && cellData.genes.length > 0) {
            console.log("First gene in first cell:", cellData.genes[0]);
          }
        }

        // Add fitness rank to cell data
        cellData.fitnessRank = sortedByFitness.indexOf(cellData);
        // Add stress level and generation from state if available
        cellData.stress_level = state.stress_level || state.stress || 0;
        cellData.generation = state.generation || 0;

        const cellId =
          cellData.cell_id || cellData.cellId || cellData.id || `cell_${index}`;
        let cell = this.cells.get(cellId);

        if (!cell) {
          // Debug: Check if this cell has S genes before creating
          const sGenes = cellData.genes
            ? cellData.genes.filter((g) => g.gene_type === "S")
            : [];
          if (sGenes.length > 0) {
            console.log(
              `Creating cell ${cellId} with ${sGenes.length} S genes`
            );
          }

          cell = this.createCellMechanism(cellData);
          this.cells.set(cellId, cell);
          this.population_mechanism.add(cell);

          // Check if cell has quantum genes and create ghost
          const hasQuantumGenes = cellData.genes?.some((g) => g.is_quantum);
          if (hasQuantumGenes) {
            const ghost = this.quantumVisualizer.createQuantumGhost(
              cell,
              cellData
            );
            this.population_mechanism.add(ghost);
          }
        }

        // Get sophisticated position based on cell properties
        const targetPosition = this.layout.assignCellPosition(
          cellData,
          index,
          state.cells
        );

        // Smooth movement to new position
        if (cell.userData.targetPosition) {
          cell.position.lerp(targetPosition, 0.05);
        } else {
          cell.position.copy(targetPosition);
        }
        cell.userData.targetPosition = targetPosition;

        // Update appearance
        this.updateCellAppearance(cell, cellData);
      });

      // Add connections between related cells
      this.updateCellConnections(state);

      // Log summary of cell types
      const cellTypeCounts = { V: 0, D: 0, J: 0, Q: 0, S: 0, balanced: 0 };
      this.cells.forEach((cell, cellId) => {
        const specialization = cell.userData.specialization;
        if (specialization && specialization.primary) {
          cellTypeCounts[specialization.primary] =
            (cellTypeCounts[specialization.primary] || 0) + 1;
        }
      });
      console.log("Cell type distribution:", cellTypeCounts);
      console.log(`Total cells created: ${this.cells.size}`);
    } else {
      console.log("No cells in state data!");
    }
  }

  updateCellConnections(state) {
    // Visualize horizontal gene transfer connections
    if (state.plasmid_transfers) {
      state.plasmid_transfers.forEach((transfer) => {
        const donor = this.cells.get(transfer.donor_id);
        const recipient = this.cells.get(transfer.recipient_id);

        if (donor && recipient) {
          this.createPlasmidBeam(donor, recipient, transfer);
        }
      });
    }

    // Visualize lineage connections
    if (state.lineage_data) {
      state.lineage_data.forEach((lineage) => {
        const parent = this.cells.get(lineage.parent_id);
        const child = this.cells.get(lineage.child_id);

        if (parent && child) {
          this.createLineageConnection(parent, child);
        }
      });
    }
  }

  createPlasmidBeam(donor, recipient, transfer) {
    // Create animated beam for gene transfer
    const geometry = new THREE.CylinderGeometry(0.2, 0.2, 1);
    const material = new THREE.MeshBasicMaterial({
      color: 0x00ff00,
      transparent: true,
      opacity: 0.6,
      emissive: 0x00ff00,
      emissiveIntensity: 0.5,
    });

    const beam = new THREE.Mesh(geometry, material);

    // Position and orient beam
    const start = donor.position.clone();
    const end = recipient.position.clone();
    const distance = start.distanceTo(end);
    const midpoint = start.clone().add(end).multiplyScalar(0.5);

    beam.position.copy(midpoint);
    beam.scale.y = distance;
    beam.lookAt(end);
    beam.rotateX(Math.PI / 2);

    this.scene.add(beam);

    // Animate and remove after 2 seconds
    const startTime = Date.now();
    const animate = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed > 2) {
        this.scene.remove(beam);
        return;
      }

      beam.material.opacity = 0.6 * (1 - elapsed / 2);
      beam.scale.x = beam.scale.z = 1 + Math.sin(elapsed * 10) * 0.2;

      requestAnimationFrame(animate);
    };
    animate();
  }

  createLineageConnection(parent, child) {
    // Create subtle line showing parent-child relationship
    const points = [parent.position, child.position];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.2,
    });

    const line = new THREE.Line(geometry, material);
    this.scene.add(line);

    // Fade out after 5 seconds
    setTimeout(() => {
      this.scene.remove(line);
    }, 5000);
  }

  updateCellAppearance(cell, cellData) {
    // Update cell properties based on new data
    const specialization = this.determineCellSpecialization(cellData);
    const fitness = cellData.fitness || 0.5;

    if (cell.userData.hub) {
      // Update color
      const hue = this.getSpecializationHue(specialization);
      const saturation = 0.3 + fitness * 0.7;
      const lightness = 0.3 + fitness * 0.2;

      // Update both color and emissive
      cell.userData.hub.material.color.setHSL(hue, saturation, lightness);
      cell.userData.hub.material.emissive.setHSL(hue, 0.8, 0.3);
      cell.userData.hub.material.emissiveIntensity = fitness * 0.2;

      // Force material update
      cell.userData.hub.material.needsUpdate = true;

      // Update size
      const targetScale = 0.8 + fitness * 0.4 + specialization.complexity * 0.2;
      cell.scale.setScalar(targetScale);
    }

    // Update gene indicators
    if (cell.userData.indicators) {
      cell.remove(cell.userData.indicators);
    }
    this.addGeneIndicators(cell, cellData, 5);

    // Update quantum glow
    if (cell.userData.quantumGlow) {
      cell.userData.quantumGlow.material.opacity =
        specialization.quantum_ratio * 0.3;
    }
  }

  handleEvent(event) {
    switch (event.type) {
      case "gene_activation":
        this.updateGeneActivation(event.data);
        break;
      case "transposition":
        this.animateTransposition(event.data);
        this.emit("transposition", event.data);
        break;
      case "cell_structure":
        this.updateCellStructure(event.data);
        break;
      case "cell_created":
        this.createCell(event.data);
        break;
      case "dream_phase_start":
        this.enterDreamPhase(event.data.num_realities || 5);
        break;
      case "dream_phase_end":
        this.exitDreamPhase();
        break;
      case "quantum_gene_emerged":
        this.onQuantumGeneEmerged(event.data);
        break;
    }
  }

  enterDreamPhase(numRealities) {
    console.log("🌙 Entering Quantum Dream Phase...");
    this.dreamVisualizer.enterDreamPhase(numRealities);
  }

  exitDreamPhase() {
    console.log("☀️ Exiting Dream Phase...");
    this.dreamVisualizer.exitDreamPhase();
  }

  setupScene() {
    // Camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      50000 // Increased far plane for distant rendering
    );
    this.camera.position.set(50, 30, 50);
    this.camera.lookAt(0, 0, 0);

    // Renderer with optimizations disabled for full scene rendering
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      logarithmicDepthBuffer: true, // Better depth precision for large scenes
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.sortObjects = false; // Disable automatic sorting
    document.getElementById("container").appendChild(this.renderer.domElement);

    // Controls
    this.controls = new THREE.OrbitControls(
      this.camera,
      this.renderer.domElement
    );
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 10000;

    // Handle resize
    window.addEventListener("resize", () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Mouse interaction
    window.addEventListener("mousemove", (event) => {
      this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    });

    window.addEventListener("click", () => this.onMouseClick());
  }

  setupLights() {
    // Brighter ambient light
    const ambientLight = new THREE.AmbientLight(0x404060, 0.8);
    this.scene.add(ambientLight);

    // Strong key light
    const keyLight = new THREE.DirectionalLight(0x8899ff, 1.0);
    keyLight.position.set(50, 100, 50);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    this.scene.add(keyLight);

    // Fill light
    const fillLight = new THREE.DirectionalLight(0x6677ff, 0.7);
    fillLight.position.set(-50, 50, -50);
    this.scene.add(fillLight);

    // Rim light for crystal edges
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
    rimLight.position.set(0, -50, 0);
    this.scene.add(rimLight);

    // Add point lights for extra brightness
    const pointLight1 = new THREE.PointLight(0x88aaff, 0.8, 200);
    pointLight1.position.set(0, 50, 0);
    this.scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xffaa88, 0.5, 150);
    pointLight2.position.set(50, 0, 50);
    this.scene.add(pointLight2);
  }

  initializeArchitecture() {
    // Main neural architecture group
    this.neural_architecture = new THREE.Group();
    this.scene.add(this.neural_architecture);

    // Create the Neural Helix Engine spine
    this.neural_spine = this.createNeuralHelixEngine();
    this.neural_architecture.add(this.neural_spine);

    // Population mechanism group
    this.population_mechanism = new THREE.Group();
    this.scene.add(this.population_mechanism);

    // Initialize Germinal Center layout
    this.layout = new TransposableNeuralCell();
    this.layout.createZoneVisualizations(this.scene);

    // Add quantum entanglement visualizer
    this.quantumVisualizer = new QuantumEntanglementVisualizer(this.scene);

    // Add dream phase visualizer
    this.dreamVisualizer = new QuantumDreamPhaseVisualizer(
      this.scene,
      this.camera
    );

    // Grid removed for cleaner visualization
  }

  createNeuralHelixEngine() {
    const helixEngine = new THREE.Group();

    // Central crystalline spine
    const spineGeometry = new THREE.CylinderGeometry(2, 2, 100, 8);
    const spineMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x2244aa,
      metalness: 0.5,
      roughness: 0.1,
      transmission: 0.9,
      thickness: 2,
      clearcoat: 1,
      clearcoatRoughness: 0,
      emissive: 0x1122aa,
      emissiveIntensity: 0.1,
    });

    const spine = new THREE.Mesh(spineGeometry, spineMaterial);
    spine.castShadow = true;
    spine.receiveShadow = true;

    // Add mechanical mounting sockets spiraling around spine
    for (let i = 0; i < 50; i++) {
      const t = i / 50;
      const angle = t * Math.PI * 8; // 4 full rotations
      const y = (t - 0.5) * 100;

      const socket = this.createMountingSocket();
      socket.position.set(Math.cos(angle) * 3, y, Math.sin(angle) * 3);
      socket.rotation.z = angle;
      spine.add(socket);
    }

    helixEngine.add(spine);
    helixEngine.userData.spine = spine;

    return helixEngine;
  }

  createMountingSocket() {
    const socketGeometry = new THREE.CylinderGeometry(0.5, 0.7, 0.5, 6);
    const socketMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x888888,
      metalness: 0.9,
      roughness: 0.2,
      clearcoat: 1,
    });

    const socket = new THREE.Mesh(socketGeometry, socketMaterial);
    socket.rotation.x = Math.PI / 2;

    return socket;
  }

  createCrystallineGeneModule(geneData) {
    const module = new THREE.Group();
    module.userData = geneData;

    // Outer crystal shell shape based on gene type
    let shellGeometry;
    switch (geneData.gene_type) {
      case "V":
        shellGeometry = new THREE.CylinderGeometry(1.5, 1.5, 3, 6); // Hexagonal
        break;
      case "D":
        shellGeometry = new THREE.CylinderGeometry(1.5, 1.5, 3, 8); // Octagonal
        break;
      case "J":
        shellGeometry = new THREE.CylinderGeometry(1.5, 1.5, 3, 12); // Dodecagonal
        break;
      case "Q":
        shellGeometry = this.createQuantumTesseract(); // 4D projection
        break;
      default:
        shellGeometry = new THREE.OctahedronGeometry(2, 0);
    }

    // Semi-transparent crystal material
    const shellMaterial = new THREE.MeshPhysicalMaterial({
      color: this.gene_type_colors[geneData.gene_type] || 0x00ffff,
      metalness: 0.2,
      roughness: 0.1,
      transmission: 0.7,
      thickness: 1,
      side: THREE.DoubleSide,
      clearcoat: 1,
      clearcoatRoughness: 0,
      emissive: this.gene_type_colors[geneData.gene_type] || 0x00ffff,
      emissiveIntensity: 0.1,
    });

    const shell = new THREE.Mesh(shellGeometry, shellMaterial);
    shell.castShadow = true;
    shell.receiveShadow = true;

    // Internal mechanism visible through crystal
    const internals = this.createInternalMechanism(geneData);
    shell.add(internals);

    module.add(shell);
    module.userData.shell = shell;
    module.userData.internals = internals;

    // Store in genes map
    this.genes.set(geneData.gene_id, module);

    return module;
  }

  createInternalMechanism(geneData) {
    const mechanism = new THREE.Group();

    // Neural ODE visualized as spinning gears
    const depth = geneData.depth || 3;
    const gears = [];

    for (let i = 0; i < Math.floor(depth * 3); i++) {
      const gearRadius = 1.5 - i * 0.2;
      if (gearRadius <= 0.3) break;

      const gear = this.createGear(gearRadius, 0.15, 8 - i);
      gear.material = new THREE.MeshPhysicalMaterial({
        color: 0xffaa00,
        metalness: 0.9,
        roughness: 0.3,
        emissive: 0xff6600,
        emissiveIntensity: 0.2,
      });

      gear.position.y = i * 0.5 - depth * 0.25;
      gear.userData.spinSpeed = 0.01 + i * 0.005;
      gear.userData.baseSpeed = gear.userData.spinSpeed;

      gears.push(gear);
      mechanism.add(gear);
    }

    // Energy conduit (glowing spiral)
    const conduit = this.createEnergyConduit(geneData.hidden_dim || 64);
    mechanism.add(conduit);

    mechanism.userData.gears = gears;
    mechanism.userData.conduit = conduit;

    return mechanism;
  }

  createGear(radius, thickness, teeth) {
    const shape = new THREE.Shape();
    const angleStep = (Math.PI * 2) / teeth;

    for (let i = 0; i < teeth; i++) {
      const angle = i * angleStep;
      const nextAngle = (i + 1) * angleStep;

      // Inner radius
      const ri = radius * 0.7;
      // Outer radius
      const ro = radius;

      // Create tooth
      const x1 = Math.cos(angle) * ri;
      const y1 = Math.sin(angle) * ri;
      const x2 = Math.cos(angle + angleStep * 0.3) * ro;
      const y2 = Math.sin(angle + angleStep * 0.3) * ro;
      const x3 = Math.cos(angle + angleStep * 0.7) * ro;
      const y3 = Math.sin(angle + angleStep * 0.7) * ro;
      const x4 = Math.cos(nextAngle) * ri;
      const y4 = Math.sin(nextAngle) * ri;

      if (i === 0) shape.moveTo(x1, y1);
      else shape.lineTo(x1, y1);

      shape.lineTo(x2, y2);
      shape.lineTo(x3, y3);
      shape.lineTo(x4, y4);
    }

    const extrudeSettings = {
      depth: thickness,
      bevelEnabled: true,
      bevelThickness: thickness * 0.1,
      bevelSize: thickness * 0.1,
      bevelSegments: 2,
    };

    const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geometry.center();

    return new THREE.Mesh(geometry);
  }

  createEnergyConduit(hidden_dim) {
    const conduit = new THREE.Group();

    // Glowing helix showing data flow
    const points = [];
    const segments = 50;

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const angle = t * Math.PI * 4;
      const radius = 0.5 + Math.sin(t * Math.PI) * 0.2;

      points.push(
        new THREE.Vector3(
          Math.cos(angle) * radius,
          (t - 0.5) * 4,
          Math.sin(angle) * radius
        )
      );
    }

    const tubeGeometry = new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(points),
      segments,
      0.05,
      8,
      false
    );

    const tubeMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x00ffff,
      emissive: 0x00ffff,
      emissiveIntensity: 0.5,
      metalness: 0.5,
      roughness: 0.1,
      transmission: 0.5,
      thickness: 0.1,
    });

    const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
    conduit.add(tube);

    // Energy particles flowing through conduit
    const particleCount = Math.floor(hidden_dim / 8);
    const particles = [];

    for (let i = 0; i < particleCount; i++) {
      const particle = new THREE.Mesh(
        new THREE.SphereGeometry(0.1, 8, 8),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.8,
        })
      );

      particle.userData.offset = i / particleCount;
      particles.push(particle);
      conduit.add(particle);
    }

    conduit.userData.particles = particles;
    conduit.userData.curve = new THREE.CatmullRomCurve3(points);

    return conduit;
  }

  createQuantumTesseract() {
    // 4D hypercube projection into 3D
    const vertices = [];
    const size = 1.5;

    // Generate 16 vertices of a tesseract
    for (let i = 0; i < 16; i++) {
      const x = i & 1 ? size : -size;
      const y = i & 2 ? size : -size;
      const z = i & 4 ? size : -size;
      const w = i & 8 ? size * 0.5 : -size * 0.5;

      // Project 4D to 3D
      const scale = 1 / (2 - w / size);
      vertices.push(new THREE.Vector3(x * scale, y * scale, z * scale));
    }

    // Create geometry from vertices
    const geometry = new THREE.BufferGeometry();
    const positions = [];

    // Connect vertices to form tesseract edges
    const edges = [
      [0, 1],
      [0, 2],
      [0, 4],
      [0, 8],
      [1, 3],
      [1, 5],
      [1, 9],
      [2, 3],
      [2, 6],
      [2, 10],
      [3, 7],
      [3, 11],
      [4, 5],
      [4, 6],
      [4, 12],
      [5, 7],
      [5, 13],
      [6, 7],
      [6, 14],
      [7, 15],
      [8, 9],
      [8, 10],
      [8, 12],
      [9, 11],
      [9, 13],
      [10, 11],
      [10, 14],
      [11, 15],
      [12, 13],
      [12, 14],
      [13, 15],
      [14, 15],
    ];

    edges.forEach((edge) => {
      positions.push(
        vertices[edge[0]].x,
        vertices[edge[0]].y,
        vertices[edge[0]].z,
        vertices[edge[1]].x,
        vertices[edge[1]].y,
        vertices[edge[1]].z
      );
    });

    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3)
    );

    return new THREE.ConvexGeometry(vertices);
  }

  createCellMechanism(cellData) {
    const cellGroup = new THREE.Group();
    cellGroup.userData = cellData;

    // Determine cell specialization based on gene composition
    const specialization = this.determineCellSpecialization(cellData);

    // Debug logging AFTER specialization is calculated
    console.log(
      `Cell ${cellData.cell_id || "unknown"}: type=${
        specialization.primary
      }, genes=${JSON.stringify(
        specialization.geneTypes
      )}, hue=${this.getSpecializationHue(specialization)}`
    );

    // Cell size based on gene count - much more dramatic variation
    const geneCount = cellData.genes ? cellData.genes.length : 3;
    // Exponential scaling for dramatic size differences
    const minSize = 1.5; // Minimum size for cells with few genes
    const maxSize = 15; // Maximum size for cells with many genes
    const normalizedGeneCount = Math.min(geneCount / 20, 1); // Normalize to 0-1 range (assuming max ~20 genes)
    // Use power function for more dramatic scaling
    const cellSize =
      minSize + (maxSize - minSize) * Math.pow(normalizedGeneCount, 0.7);

    // Cell shape based on specialization
    let hubGeometry;
    switch (specialization.primary) {
      case "V":
        // V-specialized cells are more angular
        hubGeometry = new THREE.IcosahedronGeometry(cellSize, 0);
        break;
      case "D":
        // D-specialized cells are more rounded
        hubGeometry = new THREE.SphereGeometry(cellSize, 32, 16);
        break;
      case "J":
        // J-specialized cells are elongated
        hubGeometry = new THREE.CylinderGeometry(
          cellSize * 0.8,
          cellSize,
          cellSize * 1.5,
          8
        );
        break;
      case "Q":
        // Quantum cells are complex polyhedra
        hubGeometry = new THREE.DodecahedronGeometry(cellSize, 0);
        break;
      case "S":
        // Stem cells are angelic - special handling below
        console.log(
          `Creating angelic stem cell for ${cellData.cell_id || "unknown"}`
        );
        return this.createAngelicStemCell(cellData, cellSize);
      default:
        // Balanced cells
        hubGeometry = new THREE.OctahedronGeometry(cellSize, 0);
    }

    // Color based on fitness and specialization
    const fitness = cellData.fitness || 0.5;
    const hue = this.getSpecializationHue(specialization);
    const saturation = 0.3 + fitness * 0.7;
    const lightness = 0.3 + fitness * 0.2;

    // Debug logging for color mapping (moved after specialization calculation)
    // Moved to after specialization is calculated

    const hubMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color().setHSL(hue, saturation, lightness),
      metalness: 0.3 + specialization.quantum_ratio * 0.5,
      roughness: 0.5 - fitness * 0.3,
      transmission: 0.3 + specialization.complexity * 0.3,
      clearcoat: fitness,
      clearcoatRoughness: 0.1,
      emissive: new THREE.Color().setHSL(hue, 0.8, 0.3),
      emissiveIntensity: fitness * 0.2,
    });

    const hub = new THREE.Mesh(hubGeometry, hubMaterial);
    hub.castShadow = true;
    hub.receiveShadow = true;
    cellGroup.add(hub);

    // Internal clockwork visible through hub
    const clockwork = this.createClockworkMechanism();
    clockwork.scale.setScalar(0.7);
    hub.add(clockwork);

    // Add visual gene indicators around the cell
    this.addGeneIndicators(cellGroup, cellData, cellSize);

    // Add specialization glow
    if (specialization.quantum_ratio > 0) {
      this.addQuantumGlow(cellGroup, specialization.quantum_ratio, cellSize);
    }

    // Add fitness aura
    this.addFitnessAura(cellGroup, fitness, cellSize);

    cellGroup.userData.hub = hub;
    cellGroup.userData.clockwork = clockwork;
    cellGroup.userData.gene_modules = [];
    cellGroup.userData.specialization = specialization;
    cellGroup.userData.fitness = fitness;

    // Don't set random position - will be set by caller
    this.cells.set(cellData.cell_id, cellGroup);
    this.population_mechanism.add(cellGroup);

    return cellGroup;
  }

  determineCellSpecialization(cellData) {
    const geneTypes = { V: 0, D: 0, J: 0, Q: 0, S: 0 };
    let totalGenes = 0;
    let activeGenes = 0;
    let quantumGenes = 0;

    // Check if this cell has any S genes
    let hasStemGenes = false;

    // Debug: log the cell data structure
    if (cellData.genes && cellData.genes.length > 0) {
      console.log(
        "Cell ID:",
        cellData.cell_id,
        "Sample gene structure:",
        cellData.genes[0]
      );
      console.log("Total genes in cell:", cellData.genes.length);

      // Check for S genes specifically
      const sGenes = cellData.genes.filter((gene) => gene.gene_type === "S");
      if (sGenes.length > 0) {
        console.log(
          "Found",
          sGenes.length,
          "S genes in cell:",
          cellData.cell_id
        );
      }
    } else {
      console.log(
        "Cell ID:",
        cellData.cell_id,
        "has no genes or genes array is empty"
      );
    }

    if (cellData.genes && Array.isArray(cellData.genes)) {
      cellData.genes.forEach((gene) => {
        totalGenes++;

        // Check different possible property names
        const isActive = gene.is_active !== undefined ? gene.is_active : true;
        const isQuantum = gene.is_quantum || false;
        const geneType = gene.gene_type || "V";

        if (isActive) {
          activeGenes++;

          if (isQuantum) {
            quantumGenes++;
            geneTypes["Q"] = (geneTypes["Q"] || 0) + 1;
          } else if (geneType in geneTypes) {
            geneTypes[geneType]++;
            if (geneType === "S") {
              hasStemGenes = true;
              console.log("Found stem gene in cell:", cellData.cell_id);
            }
          }
        }
      });
    }

    // Determine primary specialization
    let primary = "balanced";
    let maxCount = 0;
    let totalTypedGenes = Object.values(geneTypes).reduce((a, b) => a + b, 0);

    // Special handling for stem cells - they get priority if they have ANY S genes
    if (geneTypes.S > 0) {
      primary = "S";
      maxCount = geneTypes.S;
      console.log(
        `Cell ${cellData.cell_id} has ${geneTypes.S} S genes - classified as STEM CELL`
      );
    } else {
      // Check if any other gene type dominates (more than 40% of active genes)
      for (const [type, count] of Object.entries(geneTypes)) {
        if (type !== "S" && count > maxCount && count > totalTypedGenes * 0.4) {
          maxCount = count;
          primary = type;
        }
      }

      // If no clear specialization, consider it balanced
      if (maxCount <= totalTypedGenes * 0.4) {
        primary = "balanced";
      }
    }

    // Debug: log gene type analysis
    console.log(`Cell ${cellData.cell_id} gene analysis:`, {
      geneTypes,
      totalTypedGenes,
      maxCount,
      primary,
      threshold: totalTypedGenes * 0.4,
    });

    // Debug: log specialization determination
    if (hasStemGenes) {
      console.log(
        "Cell",
        cellData.cell_id,
        "has stem genes. Gene counts:",
        geneTypes,
        "Primary:",
        primary
      );
    }

    // Calculate diversity and complexity
    const diversity = Object.values(geneTypes).filter((c) => c > 0).length / 4;
    const complexity = totalGenes / 10;

    return {
      primary,
      geneTypes,
      diversity,
      complexity,
      quantum_ratio: quantumGenes / Math.max(activeGenes, 1),
      active_ratio: activeGenes / Math.max(totalGenes, 1),
    };
  }

  getSpecializationHue(specialization) {
    const hueMap = {
      V: 0.58, // Blue (0080ff) - Variable genes
      D: 0.33, // Green (00ff80) - Diversity genes
      J: 0.08, // Orange (ffaa00) - Joining genes
      Q: 0.83, // Magenta (ff00ff) - Quantum genes
      S: 0.0, // Pure white for stem cells (will be handled specially)
      balanced: 0.5, // Cyan (00ffff) - Balanced cells
    };
    return hueMap[specialization.primary] || 0.5;
  }

  addGeneIndicators(cellGroup, cellData, cellSize) {
    if (!cellData.genes) return;

    const indicators = new THREE.Group();

    cellData.genes.forEach((gene, index) => {
      if (!gene.is_active) return;

      // Mini crystal for each gene
      const indicatorSize = 0.5 + gene.depth * 0.2;
      let geometry;

      switch (gene.gene_type) {
        case "V":
          geometry = new THREE.TetrahedronGeometry(indicatorSize);
          break;
        case "D":
          geometry = new THREE.OctahedronGeometry(indicatorSize);
          break;
        case "J":
          geometry = new THREE.IcosahedronGeometry(indicatorSize);
          break;
        case "Q":
          geometry = new THREE.TorusKnotGeometry(
            indicatorSize * 0.7,
            indicatorSize * 0.2,
            16,
            8
          );
          break;
        case "S":
          // Stem genes get star shapes
          const starShape = new THREE.Shape();
          const starSize = indicatorSize;
          const spikes = 8;

          for (let i = 0; i < spikes * 2; i++) {
            const angle = (i / (spikes * 2)) * Math.PI * 2;
            const radius = i % 2 === 0 ? starSize : starSize * 0.5;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            if (i === 0) starShape.moveTo(x, y);
            else starShape.lineTo(x, y);
          }
          starShape.closePath();

          geometry = new THREE.ExtrudeGeometry(starShape, {
            depth: indicatorSize * 0.2,
            bevelEnabled: true,
            bevelThickness: 0.02,
            bevelSize: 0.02,
            bevelSegments: 2,
          });
          break;
      }

      // Use quantum color if it's a quantum gene, otherwise use gene type color
      const geneColor = gene.is_quantum
        ? this.gene_type_colors["Q"]
        : this.gene_type_colors[gene.gene_type] || this.gene_type_colors["V"];

      const material = new THREE.MeshPhysicalMaterial({
        color: geneColor,
        metalness: 0.8,
        roughness: 0.2,
        emissive: geneColor,
        emissiveIntensity: gene.activation || 0.2,
        transparent: true,
        opacity: 0.8,
      });

      const indicator = new THREE.Mesh(geometry, material);

      // Position in orbit around cell
      const angle = (index / cellData.genes.length) * Math.PI * 2;
      const radius = cellSize + 2 + gene.position * 2;
      const height = Math.sin(index * 0.5) * 2;

      indicator.position.set(
        Math.cos(angle) * radius,
        height,
        Math.sin(angle) * radius
      );

      indicators.add(indicator);
    });

    cellGroup.add(indicators);
    cellGroup.userData.indicators = indicators;
  }

  addQuantumGlow(cellGroup, quantumRatio, cellSize) {
    const glowGeometry = new THREE.SphereGeometry(cellSize * 1.2, 16, 16);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: 0xff00ff,
      transparent: true,
      opacity: quantumRatio * 0.3,
      blending: THREE.AdditiveBlending,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    cellGroup.add(glow);
    cellGroup.userData.quantumGlow = glow;
  }

  createAngelicStemCell(cellData, cellSize) {
    const cellGroup = new THREE.Group();
    cellGroup.userData = cellData;

    // Central divine orb - pure white, self-illuminating
    const orbGeometry = new THREE.SphereGeometry(cellSize * 0.8, 64, 32);
    const orbMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.8,
      metalness: 0.1,
      roughness: 0.2,
      transmission: 0.7,
      thickness: 0.5,
      clearcoat: 1.0,
      clearcoatRoughness: 0.0,
      transparent: true,
      opacity: 0.9,
    });

    const orb = new THREE.Mesh(orbGeometry, orbMaterial);
    cellGroup.add(orb);
    cellGroup.userData.hub = orb; // Store reference for color updates

    // Add inner light source
    const light = new THREE.PointLight(0xffffff, 2, cellSize * 5);
    light.position.set(0, 0, 0);
    cellGroup.add(light);

    // Create rotating halos
    for (let i = 0; i < 3; i++) {
      const haloRadius = cellSize * (1.2 + i * 0.3);
      const haloGeometry = new THREE.TorusGeometry(haloRadius, 0.1, 8, 32);
      const haloMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.6 - i * 0.15,
        blending: THREE.AdditiveBlending,
      });

      const halo = new THREE.Mesh(haloGeometry, haloMaterial);

      // Different rotation axes for each halo
      if (i === 0) halo.rotation.x = Math.PI / 2;
      else if (i === 1) halo.rotation.y = Math.PI / 3;
      else halo.rotation.z = Math.PI / 4;

      cellGroup.add(halo);
      cellGroup.userData[`halo${i}`] = halo;
    }

    // Create ethereal wings using particle systems
    const wingParticleCount = 200;
    const wingGeometry = new THREE.BufferGeometry();
    const wingPositions = new Float32Array(wingParticleCount * 3);
    const wingColors = new Float32Array(wingParticleCount * 3);
    const wingSizes = new Float32Array(wingParticleCount);

    for (let i = 0; i < wingParticleCount; i++) {
      // Wing shape distribution
      const wingSpan = cellSize * 2.5;
      const wingHeight = cellSize * 1.5;

      // Create wing-like distribution
      const t = i / wingParticleCount;
      const side = i < wingParticleCount / 2 ? -1 : 1;
      const wingX =
        side * (0.5 + Math.random() * 0.5) * wingSpan * Math.sin(t * Math.PI);
      const wingY =
        (Math.random() - 0.5) * wingHeight * Math.cos(t * Math.PI * 0.5);
      const wingZ = Math.random() * cellSize * 0.3;

      wingPositions[i * 3] = wingX;
      wingPositions[i * 3 + 1] = wingY;
      wingPositions[i * 3 + 2] = wingZ;

      // White to pale blue gradient
      wingColors[i * 3] = 0.9 + Math.random() * 0.1;
      wingColors[i * 3 + 1] = 0.9 + Math.random() * 0.1;
      wingColors[i * 3 + 2] = 1.0;

      wingSizes[i] = 0.1 + Math.random() * 0.2;
    }

    wingGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(wingPositions, 3)
    );
    wingGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(wingColors, 3)
    );
    wingGeometry.setAttribute("size", new THREE.BufferAttribute(wingSizes, 1));

    const wingMaterial = new THREE.PointsMaterial({
      size: 0.15,
      transparent: true,
      opacity: 0.4,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const wings = new THREE.Points(wingGeometry, wingMaterial);
    cellGroup.add(wings);
    cellGroup.userData.wings = wings;

    // Add sacred geometry - interlocking triangles
    const sacredGroup = new THREE.Group();
    for (let j = 0; j < 6; j++) {
      const angle = (j / 6) * Math.PI * 2;
      const triangleShape = new THREE.Shape();
      const size = cellSize * 0.3;

      triangleShape.moveTo(0, size);
      triangleShape.lineTo(-size * 0.866, -size * 0.5);
      triangleShape.lineTo(size * 0.866, -size * 0.5);
      triangleShape.closePath();

      const triangleGeometry = new THREE.ShapeGeometry(triangleShape);
      const triangleMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
      });

      const triangle = new THREE.Mesh(triangleGeometry, triangleMaterial);
      triangle.position.set(
        Math.cos(angle) * cellSize * 1.5,
        Math.sin(angle) * cellSize * 1.5,
        0
      );
      triangle.lookAt(0, 0, 0);

      sacredGroup.add(triangle);
    }

    cellGroup.add(sacredGroup);
    cellGroup.userData.sacredGeometry = sacredGroup;

    // Add divine particle aura
    this.addDivineAura(cellGroup, cellSize);

    // Store update function for animation
    cellGroup.userData.updateAngelic = (time) => {
      // Breathing effect on orb
      orb.scale.setScalar(1 + 0.05 * Math.sin(time * 0.001));

      // Rotate halos
      if (cellGroup.userData.halo0)
        cellGroup.userData.halo0.rotation.z += 0.002;
      if (cellGroup.userData.halo1)
        cellGroup.userData.halo1.rotation.x += 0.003;
      if (cellGroup.userData.halo2)
        cellGroup.userData.halo2.rotation.y += 0.001;

      // Gentle wing flutter
      if (wings.material.size) {
        wings.material.size = 0.15 + 0.05 * Math.sin(time * 0.002);
      }

      // Rotate sacred geometry
      if (sacredGroup) {
        sacredGroup.rotation.z += 0.0005;
      }
    };

    return cellGroup;
  }

  addDivineAura(cellGroup, cellSize) {
    const auraParticleCount = 100;
    const auraGeometry = new THREE.BufferGeometry();
    const auraPositions = new Float32Array(auraParticleCount * 3);

    for (let i = 0; i < auraParticleCount; i++) {
      // Spherical distribution
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = cellSize * (1.5 + Math.random() * 1.5);

      auraPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      auraPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      auraPositions[i * 3 + 2] = radius * Math.cos(phi);
    }

    auraGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(auraPositions, 3)
    );

    const auraMaterial = new THREE.PointsMaterial({
      color: 0xffffcc,
      size: 0.1,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const aura = new THREE.Points(auraGeometry, auraMaterial);
    cellGroup.add(aura);
    cellGroup.userData.divineAura = aura;
  }

  triggerStressEvent(stressLevel) {
    console.log("‼️‼️ HIGH STRESS EVENT TRIGGERED! ‼️‼️");

    // Store original states
    const originalStates = new Map();

    // Phase 1: System shrinks and dims
    this.cells.forEach((cell, cellId) => {
      originalStates.set(cellId, {
        scale: cell.scale.x,
        opacity: cell.userData.hub ? cell.userData.hub.material.opacity : 1,
        emissive: cell.userData.hub
          ? cell.userData.hub.material.emissive.getHex()
          : 0,
      });

      // Animate shrinking
      const shrinkTween = new TWEEN.Tween(cell.scale)
        .to({ x: 0.5, y: 0.5, z: 0.5 }, 2000)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .start();

      // Dim the cell
      if (cell.userData.hub) {
        const dimTween = new TWEEN.Tween(cell.userData.hub.material)
          .to(
            {
              opacity: 0.3,
              emissiveIntensity: 0.1,
            },
            2000
          )
          .start();
      }
    });

    // Dim the zone lights
    this.scene.traverse((obj) => {
      if (obj.isLight && obj.intensity > 0.5) {
        const dimLight = new TWEEN.Tween(obj)
          .to({ intensity: obj.intensity * 0.3 }, 2000)
          .start();
      }
    });

    // Phase 2: After 2 seconds, emergence from dark zone
    setTimeout(() => {
      // Create bright emergence effect at dark zone
      const darkZone = this.layout.zones.darkZone;
      this.createQuantumEmergence(darkZone.center);

      // Restore system gradually
      setTimeout(() => {
        this.cells.forEach((cell, cellId) => {
          const original = originalStates.get(cellId);

          // Restore scale
          const growTween = new TWEEN.Tween(cell.scale)
            .to(
              { x: original.scale, y: original.scale, z: original.scale },
              3000
            )
            .easing(TWEEN.Easing.Elastic.Out)
            .start();

          // Restore brightness
          if (cell.userData.hub && original) {
            const brightTween = new TWEEN.Tween(cell.userData.hub.material)
              .to(
                {
                  opacity: original.opacity,
                  emissiveIntensity: 0.2,
                },
                3000
              )
              .start();
          }
        });

        // Restore lights
        this.scene.traverse((obj) => {
          if (obj.isLight) {
            const restoreLight = new TWEEN.Tween(obj)
              .to({ intensity: 1.0 }, 3000)
              .start();
          }
        });
      }, 1000);
    }, 2000);
  }

  createQuantumEmergence(position) {
    // Create expanding sphere of bright particles
    const particleCount = 500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      // Start at center of dark zone
      positions[i * 3] = position.x;
      positions[i * 3 + 1] = position.y;
      positions[i * 3 + 2] = position.z;

      // Bright colors - white to cyan to blue
      const t = i / particleCount;
      colors[i * 3] = 0.5 + t * 0.5;
      colors[i * 3 + 1] = 0.8 + t * 0.2;
      colors[i * 3 + 2] = 1.0;

      sizes[i] = Math.random() * 0.5 + 0.5;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    const material = new THREE.PointsMaterial({
      size: 1.0,
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const particles = new THREE.Points(geometry, material);
    this.scene.add(particles);

    // Animate explosion outward
    const startTime = Date.now();
    const animate = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed > 5) {
        this.scene.remove(particles);
        return;
      }

      const positions = geometry.attributes.position.array;
      for (let i = 0; i < particleCount; i++) {
        // Random direction for each particle
        const theta = (i * 0.618 * Math.PI * 2) % (Math.PI * 2);
        const phi = Math.acos(2 * (i / particleCount - 0.5));

        const speed = 20 + Math.random() * 20;
        const radius = elapsed * speed;

        positions[i * 3] =
          position.x + radius * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] =
          position.y + radius * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = position.z + radius * Math.cos(phi);
      }

      geometry.attributes.position.needsUpdate = true;
      material.opacity = 1.0 - elapsed / 5;

      requestAnimationFrame(animate);
    };
    animate();

    // Flash message
    console.log("✨ New cells emerging from the Dark Zone! ✨");
  }

  onQuantumGeneEmerged(data) {
    console.log("✨✨ A Quantum Gene has emerged! ✨✨");

    // Find the central core pipe
    let centralCore = null;
    this.scene.traverse((obj) => {
      if (obj.userData && obj.userData.isCentralCore) {
        centralCore = obj;
      }
    });

    // If no central core exists, create one
    if (!centralCore) {
      const coreGeometry = new THREE.CylinderGeometry(5, 5, 100, 32, 1, true);
      const coreMaterial = new THREE.MeshPhysicalMaterial({
        color: 0x4444ff,
        emissive: 0x2222ff,
        emissiveIntensity: 0.5,
        metalness: 0.9,
        roughness: 0.1,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });

      centralCore = new THREE.Mesh(coreGeometry, coreMaterial);
      centralCore.position.set(0, 0, 0);
      centralCore.userData.isCentralCore = true;
      this.scene.add(centralCore);
    }

    // Store original color
    const originalColor = centralCore.material.color.getHex();
    const originalEmissive = centralCore.material.emissive.getHex();

    // Flash bright pink
    const flashSequence = [
      { time: 0, color: 0xff00ff, emissive: 0xff00ff, intensity: 2.0 },
      { time: 300, color: 0xff88ff, emissive: 0xff88ff, intensity: 3.0 },
      { time: 600, color: 0xffffff, emissive: 0xffffff, intensity: 4.0 },
      { time: 900, color: 0xff88ff, emissive: 0xff88ff, intensity: 3.0 },
      { time: 1200, color: 0xff00ff, emissive: 0xff00ff, intensity: 2.0 },
      {
        time: 1800,
        color: originalColor,
        emissive: originalEmissive,
        intensity: 0.5,
      },
    ];

    // Execute flash sequence
    flashSequence.forEach((step, index) => {
      setTimeout(() => {
        centralCore.material.color.setHex(step.color);
        centralCore.material.emissive.setHex(step.emissive);
        centralCore.material.emissiveIntensity = step.intensity;
        centralCore.material.needsUpdate = true;
      }, step.time);
    });

    // Create quantum ripple effect
    this.createQuantumRipple(centralCore.position);

    // Emit quantum particles from core
    this.emitQuantumParticles(centralCore.position);
  }

  createQuantumRipple(position) {
    // Create expanding ring
    const ringGeometry = new THREE.RingGeometry(0.1, 1, 64);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: 0xff00ff,
      transparent: true,
      opacity: 1.0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });

    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.position.copy(position);
    ring.rotation.x = Math.PI / 2;
    this.scene.add(ring);

    // Animate expansion
    const startTime = Date.now();
    const animate = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed > 3) {
        this.scene.remove(ring);
        return;
      }

      const scale = 1 + elapsed * 30;
      ring.scale.set(scale, scale, 1);
      ring.material.opacity = 1.0 - elapsed / 3;

      requestAnimationFrame(animate);
    };
    animate();
  }

  emitQuantumParticles(position) {
    // Create quantum particles spiraling outward
    const particleCount = 200;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = position.x;
      positions[i * 3 + 1] = position.y;
      positions[i * 3 + 2] = position.z;

      // Pink to purple gradient
      colors[i * 3] = 1.0;
      colors[i * 3 + 1] = 0.0;
      colors[i * 3 + 2] = 1.0;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const particles = new THREE.Points(geometry, material);
    this.scene.add(particles);

    // Animate spiraling motion
    const startTime = Date.now();
    const animate = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed > 4) {
        this.scene.remove(particles);
        return;
      }

      const positions = geometry.attributes.position.array;
      for (let i = 0; i < particleCount; i++) {
        const t = i / particleCount;
        const angle = t * Math.PI * 8 + elapsed * 3;
        const radius = elapsed * 15 * (1 + t);
        const height = (t - 0.5) * 50 * elapsed;

        positions[i * 3] = position.x + Math.cos(angle) * radius;
        positions[i * 3 + 1] = position.y + height;
        positions[i * 3 + 2] = position.z + Math.sin(angle) * radius;
      }

      geometry.attributes.position.needsUpdate = true;
      material.opacity = 1.0 - elapsed / 4;

      requestAnimationFrame(animate);
    };
    animate();
  }

  createTestStemCell() {
    // Create a test stem cell to verify visualization works
    const testStemData = {
      cell_id: "test-stem-001",
      fitness: 0.8,
      genes: [
        { gene_type: "S", is_active: true, is_quantum: false, depth: 0.5 },
        { gene_type: "S", is_active: true, is_quantum: false, depth: 0.7 },
        { gene_type: "S", is_active: true, is_quantum: false, depth: 0.6 },
      ],
    };

    console.log("Creating test stem cell with data:", testStemData);

    const stemCell = this.createCellMechanism(testStemData);
    stemCell.position.set(150, 50, 0); // Position it away from other cells

    if (this.population_mechanism) {
      this.population_mechanism.add(stemCell);
    } else {
      this.scene.add(stemCell);
    }

    // Store in cells map
    this.cells.set(testStemData.cell_id, stemCell);

    console.log("Test stem cell created and positioned at (150, 50, 0)");
  }

  addFitnessAura(cellGroup, fitness, cellSize) {
    const particleCount = Math.floor(fitness * 50);
    if (particleCount === 0) return;

    const particlesGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = cellSize + Math.random() * 3;

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      const color = new THREE.Color().setHSL(0.3 * fitness, 0.8, 0.6);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    particlesGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );
    particlesGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colors, 3)
    );

    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    cellGroup.add(particles);
    cellGroup.userData.fitnessAura = particles;
  }

  createClockworkMechanism() {
    const mechanism = new THREE.Group();

    // Main drive gear
    const mainGear = this.createGear(2, 0.3, 12);
    mainGear.material = new THREE.MeshPhysicalMaterial({
      color: 0xffd700,
      metalness: 1,
      roughness: 0.4,
      clearcoat: 1,
      emissive: 0x886600,
      emissiveIntensity: 0.1,
    });
    mechanism.add(mainGear);
    mechanism.userData.mainGear = mainGear;

    // Satellite gears in different planes
    const satelliteGears = [];
    for (let i = 0; i < 3; i++) {
      const angle = (i / 3) * Math.PI * 2;
      const gear = this.createGear(1, 0.2, 8);

      gear.material = mainGear.material.clone();
      gear.position.set(
        Math.cos(angle) * 2,
        Math.sin(angle) * 0.5,
        Math.sin(angle) * 2
      );

      // Different rotation axes for complex motion
      gear.rotation.set(
        Math.random() * Math.PI * 0.25,
        angle,
        Math.random() * Math.PI * 0.25
      );

      gear.userData.rotationSpeed = 0.02 * (i + 1);
      satelliteGears.push(gear);
      mechanism.add(gear);
    }

    mechanism.userData.satelliteGears = satelliteGears;

    return mechanism;
  }

  updateCellStructure(data) {
    let cell = this.cells.get(data.cell_id);

    if (!cell) {
      // Create new cell
      cell = this.createCellMechanism(data);
    }

    // Clear old gene modules
    cell.userData.gene_modules.forEach((module) => {
      cell.remove(module);
    });
    cell.userData.gene_modules = [];

    // Add gene modules in orbital arrangement
    if (data.genes) {
      data.genes.forEach((geneData, i) => {
        let geneModule = this.genes.get(geneData.gene_id);

        if (!geneModule) {
          geneModule = this.createCrystallineGeneModule(geneData);
        }

        // Position in orbit around cell
        const angle = (i / data.genes.length) * Math.PI * 2;
        const radius = 15 + (geneData.position || i) * 2;
        const height = Math.sin(angle * 2) * 5;

        geneModule.position.set(
          Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
        );

        // Orient towards center
        geneModule.lookAt(cell.position);

        cell.add(geneModule);
        cell.userData.gene_modules.push(geneModule);

        // Create neural connection beam
        this.createNeuralConnection(cell, geneModule, geneData);
      });
    }
  }

  createNeuralConnection(cell, geneModule, geneData) {
    const connectionId = `${cell.userData.cell_id}_${geneData.gene_id}`;

    // Remove old connection if exists
    const oldConnection = this.connections.get(connectionId);
    if (oldConnection) {
      cell.remove(oldConnection);
    }

    // Create glowing beam connection
    const points = [new THREE.Vector3(0, 0, 0), geneModule.position];

    const beamGeometry = new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(points),
      20,
      0.1,
      8,
      false
    );

    const beamMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x00ffff,
      emissive: 0x00ffff,
      emissiveIntensity: 0.3,
      metalness: 0.5,
      roughness: 0.1,
      transmission: 0.8,
      thickness: 0.05,
      transparent: true,
      opacity: 0.7,
    });

    const beam = new THREE.Mesh(beamGeometry, beamMaterial);
    beam.userData.geneData = geneData;

    cell.add(beam);
    this.connections.set(connectionId, beam);

    return beam;
  }

  updateGeneActivation(data) {
    const gene = this.genes.get(data.gene_id);
    if (!gene) return;

    // Update shell glow based on activation
    if (gene.userData.shell) {
      gene.userData.shell.material.emissiveIntensity =
        0.1 + data.activation * 0.4;
    }

    // Update internal gear speeds
    if (gene.userData.internals && gene.userData.internals.userData.gears) {
      gene.userData.internals.userData.gears.forEach((gear, i) => {
        gear.userData.spinSpeed =
          gear.userData.baseSpeed * (1 + data.activation * 3);
      });
    }

    // Scale based on depth
    if (data.depth) {
      const targetScale = 0.8 + data.depth * 0.2;
      gene.scale.setScalar(targetScale);
    }

    // Update quantum state if quantum gene
    if (data.is_quantum) {
      this.updateQuantumGene(gene, data);
    }
  }

  updateQuantumGene(gene, data) {
    let quantumEffect = this.quantum_genes.get(data.gene_id);

    if (!quantumEffect) {
      // Create quantum superposition effect
      quantumEffect = new THREE.Group();

      // Duplicate crystal in different phase
      const phaseShell = gene.userData.shell.clone();
      phaseShell.material = gene.userData.shell.material.clone();
      phaseShell.material.opacity = 0.3;
      phaseShell.material.transparent = true;

      quantumEffect.add(phaseShell);
      quantumEffect.userData.phaseShell = phaseShell;

      gene.add(quantumEffect);
      this.quantum_genes.set(data.gene_id, quantumEffect);
    }

    // Update phase offset
    quantumEffect.userData.phase = (quantumEffect.userData.phase || 0) + 0.05;
  }

  animateTransposition(data) {
    const gene = this.genes.get(data.gene_id);
    if (!gene) return;

    switch (data.action) {
      case "jump":
        this.animateJump(gene, data);
        break;
      case "duplicate":
        this.animateDuplication(gene, data);
        break;
      case "invert":
        this.animateInversion(gene);
        break;
      case "delete":
        this.animateDeletion(gene);
        break;
      case "quantum_leap":
        this.animateQuantumLeap(gene, data);
        break;
    }
  }

  animateJump(gene, data) {
    // Crystal detaches with mechanical precision
    const startPos = gene.position.clone();
    const endPos = new THREE.Vector3(
      (Math.random() - 0.5) * 30,
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 30
    );

    // Increase internal gear speed during jump
    if (gene.userData.internals) {
      gene.userData.internals.userData.gears.forEach((gear) => {
        gear.userData.jumpSpeed = gear.userData.spinSpeed * 5;
      });
    }

    // Smooth jump animation
    const jumpDuration = 2000;
    const startTime = Date.now();

    const animateJumpFrame = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / jumpDuration, 1);

      // Parabolic trajectory
      const height = Math.sin(progress * Math.PI) * 20;

      gene.position.lerpVectors(startPos, endPos, progress);
      gene.position.y += height;

      // Rotate during flight
      gene.rotation.x += 0.05;
      gene.rotation.y += 0.03;

      if (progress < 1) {
        requestAnimationFrame(animateJumpFrame);
      } else {
        // Reset gear speeds
        if (gene.userData.internals) {
          gene.userData.internals.userData.gears.forEach((gear) => {
            gear.userData.jumpSpeed = null;
          });
        }
      }
    };

    animateJumpFrame();
  }

  animateDuplication(gene, data) {
    // Crystal grows a twin through fractal subdivision
    const clone = gene.clone();
    clone.userData = { ...gene.userData };

    // Start small and grow
    clone.scale.setScalar(0.1);
    clone.position.copy(gene.position);
    clone.position.x += 5;

    // Add to same parent
    gene.parent.add(clone);

    // Growth animation
    const growDuration = 1500;
    const startTime = Date.now();

    const animateGrowth = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / growDuration, 1);

      clone.scale.setScalar(0.1 + progress * 0.9);
      clone.rotation.y += 0.05;

      if (progress < 1) {
        requestAnimationFrame(animateGrowth);
      }
    };

    animateGrowth();
  }

  animateInversion(gene) {
    // Internal mechanism flips while shell rotates
    const duration = 1000;
    const startTime = Date.now();
    const startRotation = gene.rotation.x;

    const animateFlip = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Flip the entire gene
      gene.rotation.x = startRotation + Math.PI * progress;

      // Counter-rotate internals for complex motion
      if (gene.userData.internals) {
        gene.userData.internals.rotation.x = -Math.PI * progress;
      }

      if (progress < 1) {
        requestAnimationFrame(animateFlip);
      }
    };

    animateFlip();
  }

  animateDeletion(gene) {
    // Crystal shatters and fades
    const duration = 1000;
    const startTime = Date.now();

    const animateFade = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Shrink and fade
      gene.scale.setScalar(1 - progress);

      if (gene.userData.shell) {
        gene.userData.shell.material.opacity = 1 - progress;
        gene.userData.shell.material.transparent = true;
      }

      if (progress < 1) {
        requestAnimationFrame(animateFade);
      } else {
        // Remove from scene
        gene.parent.remove(gene);
        this.genes.delete(gene.userData.gene_id);
      }
    };

    animateFade();
  }

  animateQuantumLeap(gene, data) {
    // Quantum phase transition with particle effects
    const particleCount = 100;
    const particles = new THREE.Group();

    for (let i = 0; i < particleCount; i++) {
      const particle = new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 4, 4),
        new THREE.MeshBasicMaterial({
          color: 0xff00ff,
          transparent: true,
          opacity: 0.8,
        })
      );

      // Random distribution around gene
      particle.position.set(
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 4
      );

      particle.userData.velocity = new THREE.Vector3(
        (Math.random() - 0.5) * 0.5,
        (Math.random() - 0.5) * 0.5,
        (Math.random() - 0.5) * 0.5
      );

      particles.add(particle);
    }

    gene.add(particles);

    // Animate quantum collapse
    const duration = 2000;
    const startTime = Date.now();

    const animateCollapse = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      particles.children.forEach((particle) => {
        // Expand outward
        particle.position.add(particle.userData.velocity);

        // Fade out
        particle.material.opacity = 0.8 * (1 - progress);
      });

      // Phase shift the gene
      if (gene.userData.shell) {
        const hue = progress * 0.3;
        gene.userData.shell.material.color.setHSL(hue, 0.8, 0.5);
      }

      if (progress < 1) {
        requestAnimationFrame(animateCollapse);
      } else {
        gene.remove(particles);
      }
    };

    animateCollapse();
  }

  updatePopulationStructure_OLD_UNUSED(state) {
    // This was creating generic cells - replaced by the proper method above
    // Display ALL B-cells in the population
    const populationSize = state.population || 512;
    const currentCellCount = this.cells.size;

    if (populationSize !== currentCellCount) {
      // Clear and recreate to match actual population
      this.cells.forEach((cell, id) => {
        this.population_mechanism.remove(cell);
      });
      this.cells.clear();

      // Create cells in a organized 3D sphere arrangement for better visualization
      const goldenRatio = (1 + Math.sqrt(5)) / 2;
      const angleIncrement = Math.PI * (3 - Math.sqrt(5)); // Golden angle

      for (let i = 0; i < populationSize; i++) {
        // Use fibonacci sphere distribution for even spacing
        const t = i / (populationSize - 1);
        const inclination = Math.acos(1 - 2 * t);
        const azimuth = angleIncrement * i;

        // Sphere radius increases with population
        const radius = 100 + Math.pow(populationSize / 100, 0.8) * 50;

        const x = radius * Math.sin(inclination) * Math.cos(azimuth);
        const y = radius * Math.cos(inclination);
        const z = radius * Math.sin(inclination) * Math.sin(azimuth);

        const cellData = {
          cell_id: `bcell_${i}`,
          position: { x, y, z },
        };

        const cell = this.createCellMechanism(cellData);
        // Set position
        cell.position.set(x, y, z);

        // Make cells smaller for large populations
        const cellScale = Math.max(0.2, 1 - populationSize / 1000);
        cell.scale.setScalar(cellScale);
      }

      // Adjust camera for large population
      if (populationSize > 100) {
        this.camera.position.set(300, 200, 300);
        this.controls.maxDistance = 1000;
      }
    }

    // Update cell states based on actual metrics
    this.cells.forEach((cell, cellId) => {
      if (cell.userData.hub) {
        // Pulse based on fitness
        const pulse =
          0.8 + Math.sin(this.time * 2) * 0.2 * (state.fitness || 0.5);
        cell.userData.hub.material.emissiveIntensity = 0.1 * pulse;

        // Color based on cell fitness (gradient from blue to green to yellow)
        const fitness = state.fitness || 0.5;
        const hue = 0.6 - fitness * 0.4; // Blue (0.6) to Green (0.3) to Yellow (0.2)
        cell.userData.hub.material.color.setHSL(hue, 0.7, 0.5);

        // Size based on diversity
        const scale = 0.8 + (state.diversity || 0) * 0.02;
        cell.scale.setScalar(scale);
      }
    });

    // Add text to show actual population count
    if (!this.populationText) {
      const loader = new THREE.FontLoader();
      // For now, just log it
      console.log(
        `Visualizing ${state.cells ? state.cells.length : 0} of ${
          state.population_size || 0
        } B-cells`
      );
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    this.time += 0.016; // ~60fps

    // Update TWEEN animations
    if (TWEEN) {
      TWEEN.update();
    }

    // Update controls
    this.controls.update();

    // Update quantum visualizations
    if (this.quantumVisualizer) {
      this.quantumVisualizer.update(this.time);
    }

    // Update dream phase
    if (this.dreamVisualizer) {
      this.dreamVisualizer.update(this.time);
    }

    // Animate genes
    this.genes.forEach((gene, geneId) => {
      // Rotate gene slowly
      gene.rotation.y += 0.005;

      // Animate internal gears
      if (gene.userData.internals && gene.userData.internals.userData.gears) {
        gene.userData.internals.userData.gears.forEach((gear) => {
          const speed =
            gear.userData.jumpSpeed || gear.userData.spinSpeed || 0.01;
          gear.rotation.z += speed;
        });
      }

      // Animate energy conduit
      if (gene.userData.internals && gene.userData.internals.userData.conduit) {
        const conduit = gene.userData.internals.userData.conduit;

        // Flow particles through conduit
        if (conduit.userData.particles && conduit.userData.curve) {
          conduit.userData.particles.forEach((particle, i) => {
            const offset = (particle.userData.offset + this.time * 0.1) % 1;
            const point = conduit.userData.curve.getPoint(offset);
            particle.position.copy(point);
          });
        }
      }
    });

    // Animate cells with personality based on their properties
    this.cells.forEach((cell, cellId) => {
      const specialization = cell.userData.specialization;

      if (specialization) {
        // Different movement patterns by specialization
        if (specialization.primary === "V") {
          // V cells are more active, darting movements
          cell.position.x +=
            Math.sin(this.time * 2 + cellId.charCodeAt(0)) * 0.1;
          cell.position.z +=
            Math.cos(this.time * 2 + cellId.charCodeAt(0)) * 0.1;
        } else if (specialization.primary === "D") {
          // D cells rotate more
          cell.rotation.y += 0.01 * specialization.diversity;
        } else if (specialization.primary === "J") {
          // J cells bob up and down
          cell.position.y +=
            Math.sin(this.time * 3 + cellId.charCodeAt(0)) * 0.05;
        } else if (specialization.primary === "Q") {
          // Quantum cells phase in and out
          if (cell.userData.hub) {
            const phase = Math.sin(this.time * 4 + cellId.charCodeAt(0));
            cell.userData.hub.material.opacity = 0.7 + phase * 0.3;
          }
        } else if (specialization.primary === "S") {
          // Angelic stem cells - special animations
          if (cell.userData.updateAngelic) {
            cell.userData.updateAngelic(this.time * 1000);
          }
        }

        // Pulse based on fitness
        const pulse =
          1 + Math.sin(this.time * 2) * 0.1 * (cell.userData.fitness || 0.5);
        const currentScale = cell.scale.x;
        cell.scale.setScalar(currentScale * pulse);
        cell.scale.multiplyScalar(1 / pulse); // Normalize back
      }

      // Rotate gene indicators
      if (cell.userData.indicators) {
        cell.userData.indicators.rotation.y += 0.005;
      }

      // Animate fitness aura
      if (cell.userData.fitnessAura) {
        cell.userData.fitnessAura.rotation.y += 0.01;
        cell.userData.fitnessAura.rotation.x += 0.005;
      }
      // Rotate clockwork mechanism
      if (cell.userData.clockwork) {
        const clockwork = cell.userData.clockwork;

        if (clockwork.userData.mainGear) {
          clockwork.userData.mainGear.rotation.z += 0.01;
        }

        if (clockwork.userData.satelliteGears) {
          clockwork.userData.satelliteGears.forEach((gear, i) => {
            gear.rotation.z += gear.userData.rotationSpeed;
            gear.rotation.y += gear.userData.rotationSpeed * 0.3;
          });
        }
      }

      // Gentle floating motion
      cell.position.y += Math.sin(this.time + cellId.charCodeAt(0)) * 0.02;
    });

    // Animate quantum genes
    this.quantum_genes.forEach((quantumEffect, geneId) => {
      if (quantumEffect.userData.phaseShell) {
        // Phase oscillation
        const phase = quantumEffect.userData.phase || 0;
        quantumEffect.userData.phaseShell.position.x = Math.sin(phase) * 0.5;
        quantumEffect.userData.phaseShell.position.z = Math.cos(phase) * 0.5;
        quantumEffect.userData.phaseShell.material.opacity =
          0.3 + Math.sin(phase * 2) * 0.2;
      }
    });

    // Update connections
    this.connections.forEach((beam, connectionId) => {
      if (beam.userData.geneData) {
        // Pulse connection based on activation
        const activation = beam.userData.geneData.activation || 0.5;
        beam.material.emissiveIntensity =
          0.3 + Math.sin(this.time * 5) * 0.2 * activation;
      }
    });

    // Quantum cells have special synchronized behavior
    this.cells.forEach((cell, cellId) => {
      if (cell.userData.specialization?.primary === "Q") {
        const ghost = this.quantumVisualizer.entangledPairs.get(cellId);
        if (ghost) {
          // Opposite rotation - perfect anti-correlation
          ghost.rotation.y = -cell.rotation.y;
          ghost.rotation.x = -cell.rotation.x;

          // Opposite phase pulsing
          const pulse = 1 + Math.sin(this.time * 4) * 0.1;
          cell.scale.setScalar(pulse);
          ghost.scale.setScalar(2 - pulse); // Inverse scaling

          // Update entanglement strength based on distance
          const distance = cell.position.distanceTo(ghost.position);
          const strength = Math.max(0, 1 - distance / 200);

          if (ghost.userData.hub?.material?.uniforms) {
            ghost.userData.hub.material.uniforms.entanglementStrength.value =
              strength;
          }
        }
      }
    });

    // Gentle rotation of entire architecture
    this.neural_architecture.rotation.y += 0.0005;
    this.population_mechanism.rotation.y -= 0.0003;

    // Render
    this.renderer.render(this.scene, this.camera);
  }

  onMouseClick() {
    // Raycast to select objects
    this.raycaster.setFromCamera(this.mouse, this.camera);

    const intersects = this.raycaster.intersectObjects(
      this.scene.children,
      true
    );

    if (intersects.length > 0) {
      const object = intersects[0].object;

      // Find the gene or cell this object belongs to
      let selected = null;
      let current = object;

      while (current && !selected) {
        if (current.userData.gene_id) {
          selected = { type: "gene", data: current.userData };
        } else if (current.userData.cell_id) {
          selected = { type: "cell", data: current.userData };
        }
        current = current.parent;
      }

      if (selected) {
        this.emit("object_selected", selected.data);
      }
    }
  }

  // Event emitter pattern
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }

  emit(event, data) {
    if (this.eventHandlers.has(event)) {
      this.eventHandlers.get(event).forEach((handler) => handler(data));
    }
  }

  // Public methods for view control
  changeView(viewName) {
    switch (viewName) {
      case "molecular":
        // Zoom into a single gene
        if (this.genes.size > 0) {
          const firstGene = this.genes.values().next().value;
          this.focusOn(firstGene, 10);
        }
        break;

      case "cellular":
        // View single cell with genes
        if (this.cells.size > 0) {
          const firstCell = this.cells.values().next().value;
          this.focusOn(firstCell, 50);
        }
        break;

      case "population":
        // View all cells
        this.camera.position.set(100, 50, 100);
        this.controls.target.set(0, 0, 0);
        break;

      case "abstract":
        // Data flow view - hide meshes, show connections
        this.toggleAbstractView();
        break;
    }
  }

  focusOn(object, distance) {
    const targetPosition = new THREE.Vector3();
    object.getWorldPosition(targetPosition);

    this.controls.target.copy(targetPosition);

    const offset = new THREE.Vector3(distance, distance * 0.5, distance);
    this.camera.position.copy(targetPosition).add(offset);

    this.controls.update();
  }

  toggleAbstractView() {
    // Toggle between physical and abstract visualization
    this.genes.forEach((gene) => {
      gene.visible = !gene.visible;
    });

    this.cells.forEach((cell) => {
      if (cell.userData.hub) {
        cell.userData.hub.visible = !cell.userData.hub.visible;
      }
    });

    // Show only connections in abstract mode
    this.connections.forEach((connection) => {
      connection.material.emissiveIntensity = connection.visible ? 0.3 : 1.0;
    });
  }

  togglePause() {
    // Pause/resume animations
    this.paused = !this.paused;
  }
}

// Global functions for HTML buttons
function changeView(view) {
  if (window.viz) {
    window.viz.changeView(view);
  }
}

function togglePause() {
  if (window.viz) {
    window.viz.togglePause();
  }
}

function toggleInfo() {
  const panel = document.getElementById("info-panel");
  if (panel) {
    panel.classList.toggle("visible");
  }
}

function resetCamera() {
  if (window.viz) {
    window.viz.camera.position.set(50, 30, 50);
    window.viz.controls.target.set(0, 0, 0);
    window.viz.controls.update();
  }
}
// FPS Counter Overlay

// Create FPS display element
const fpsDisplay = document.createElement("div");
fpsDisplay.style.position = "fixed";
fpsDisplay.style.left = "10px";
fpsDisplay.style.top = "10px";
fpsDisplay.style.background = "rgba(0,0,0,0.6)";
fpsDisplay.style.color = "#00ffcc";
fpsDisplay.style.padding = "4px 10px";
fpsDisplay.style.fontFamily = "monospace";
fpsDisplay.style.fontSize = "14px";
fpsDisplay.style.borderRadius = "6px";
fpsDisplay.style.zIndex = 10000;
fpsDisplay.style.pointerEvents = "none";
fpsDisplay.textContent = "FPS: ...";
document.body.appendChild(fpsDisplay);

let lastFrameTime = performance.now();
let frames = 0;
let lastFpsUpdate = performance.now();
let fps = 0;

function updateFPS() {
  const now = performance.now();
  frames++;
  if (now - lastFpsUpdate > 500) {
    fps = Math.round((frames * 1000) / (now - lastFpsUpdate));
    fpsDisplay.textContent = `FPS: ${fps}`;
    lastFpsUpdate = now;
    frames = 0;
  }
  requestAnimationFrame(updateFPS);
}
requestAnimationFrame(updateFPS);
