# rag_knowledge_base.py
# OSHA + Indian Construction Safety Regulations
# Used by the RAG pipeline to retrieve relevant regulations per violation

REGULATIONS = [

    # ─── HELMET / HEAD PROTECTION ───────────────────────────────────────────
    "OSHA 1926.100(a): Employees working in areas where there is a possible danger "
    "of head injury from impact, falling or flying objects, or electrical shock "
    "shall be protected by protective helmets. Violation penalty: up to $15,625 per incident.",

    "OSHA 1926.100(b): Helmets for protection against impact and penetration of "
    "falling and flying objects shall meet the requirements of ANSI Z89.1.",

    "IS 2925:1984 (India): Industrial safety helmets shall be worn at all times "
    "in construction zones. Helmets must withstand impact of 5kg mass from 1m height. "
    "Mandatory under Building and Other Construction Workers Act 1996, Section 38.",

    "Building and Other Construction Workers Act 1996, Section 38(1): Every employer "
    "shall provide helmets free of cost to every construction worker before they "
    "commence work on any construction site.",

    # ─── MASK / RESPIRATORY PROTECTION ──────────────────────────────────────
    "OSHA 1926.103: Respiratory protection shall be provided by employers when "
    "engineering and administrative controls are not feasible or while they are "
    "being implemented. Dust masks required in areas with visible particulate matter.",

    "OSHA 1910.134(c)(1): Where respirator use is required, the employer shall "
    "establish and implement a written respiratory protection program. "
    "Violation penalty: up to $15,625 per incident.",

    "IS 9473:2002 (India): Respiratory protective devices shall conform to BIS standards. "
    "Dust masks mandatory in all areas with concrete cutting, grinding, or demolition "
    "under Factories Act 1948, Section 14.",

    "NIOSH Guidelines: N95 respirators or higher required when PM2.5 levels exceed "
    "35 micrograms per cubic meter. Construction dust exposure limit: 5mg/m3 (total dust).",

    # ─── SAFETY VEST / HIGH VISIBILITY ──────────────────────────────────────
    "OSHA 1926.201: Flaggers and all workers in active traffic zones shall wear "
    "high-visibility safety vests meeting ANSI/ISEA 107 Class 2 or Class 3 requirements. "
    "Violation penalty: up to $15,625 per incident.",

    "OSHA 1910.178(l): Workers operating near forklifts, cranes, or heavy equipment "
    "must wear high-visibility vests at all times to ensure operator awareness.",

    "IS 15809:2008 (India): High visibility warning clothing mandatory for workers "
    "in areas with moving vehicles or machinery. Fluorescent yellow-green or orange-red "
    "colors required under Motor Vehicles Act for road construction workers.",

    "Construction (Design and Management) Regulations 2015, Regulation 13: "
    "Contractors must ensure all workers wear appropriate high-visibility clothing "
    "when working in zones with vehicle or plant movement.",

    # ─── GLOVES / HAND PROTECTION ───────────────────────────────────────────
    "OSHA 1926.28(a): The employer is responsible for requiring the wearing of "
    "appropriate personal protective equipment in all operations where there is "
    "an exposure to hazardous conditions including hand injuries.",

    "OSHA 1910.138(a): Employers shall select and require employees to use "
    "appropriate hand protection when employees' hands are exposed to hazards such "
    "as cuts, lacerations, chemical burns, or thermal burns. Penalty: up to $15,625.",

    "IS 6994:1973 (India): Leather safety gloves shall be worn during handling of "
    "construction materials, reinforcement steel, and sharp objects under "
    "Building and Other Construction Workers Act 1996.",

    # ─── GOGGLES / EYE PROTECTION ───────────────────────────────────────────
    "OSHA 1926.102(a)(1): Employees shall be provided with eye and face protection "
    "equipment when machines or operations present potential eye or face injury. "
    "Minimum ANSI Z87.1 standard required. Penalty: up to $15,625.",

    "OSHA 1926.102(b): Eye protection used during welding, cutting, or torch brazing "
    "shall meet the requirements specified in Table E-2 of 29 CFR 1926.102.",

    "IS 5983:1980 (India): Eye protectors shall be worn during grinding, welding, "
    "cutting, and any operation producing chips, sparks, or radiation. "
    "Mandatory under Factories Act 1948, Section 35.",

    # ─── MACHINERY PROXIMITY / EXCLUSION ZONES ──────────────────────────────
    "OSHA 1926.600(a)(1): All machinery and equipment shall be maintained in "
    "operating condition. Employees shall not operate equipment unless safe. "
    "Exclusion zones required around all active heavy machinery.",

    "OSHA 1926.502(j): Safety monitoring systems: when a safety monitoring system "
    "is used on low-slope roofs, a safety monitor shall warn workers when they "
    "appear to be unaware of a fall hazard or are acting in an unsafe manner.",

    "OSHA 1910.212(a)(1): One or more methods of machine guarding shall be provided "
    "to protect the operator and other employees in the machine area from hazards. "
    "Exclusion zone minimum 3 meters from active machinery operation.",

    "IS 7969:1975 (India): Safety code for handling and storage of building materials. "
    "No worker shall enter the operating radius of cranes or excavators without "
    "explicit authorization and PPE verification.",

    "Building and Other Construction Workers Act 1996, Section 41: Every employer "
    "shall ensure that no worker is employed within 6 meters of an excavation "
    "without edge protection and PPE compliance.",

    # ─── FALL PROTECTION ────────────────────────────────────────────────────
    "OSHA 1926.501(b)(1): Each employee on a walking/working surface with an "
    "unprotected side or edge which is 6 feet or more above a lower level shall "
    "be protected from falling by guardrails, safety nets, or personal fall arrest.",

    "OSHA 1926.502(d): Personal fall arrest systems shall be rigged so employees "
    "cannot free fall more than 6 feet or contact any lower level. "
    "Penalty for non-compliance: up to $15,625 per violation.",

    # ─── GENERAL SITE SAFETY ────────────────────────────────────────────────
    "OSHA 1926.20(a): No contractor or subcontractor shall require any laborer or "
    "mechanic employed in the performance of the contract to work in surroundings "
    "or under conditions which are unsanitary, hazardous, or dangerous to health.",

    "OSHA 1926.21(b)(2): The employer shall instruct each employee in recognition "
    "and avoidance of unsafe conditions and the regulations applicable to the "
    "work environment to control or eliminate hazards.",

    "Building and Other Construction Workers Act 1996, Section 32: Every employer "
    "shall provide and maintain a safe work environment. Failure to enforce PPE "
    "compliance is a punishable offence with fine up to Rs. 2 lakhs.",

    "National Building Code of India 2016, Part 7, Section 19.2: All construction "
    "workers shall be provided with appropriate PPE at no cost. Site supervisors "
    "are personally liable for PPE non-compliance incidents.",

    # ─── REPEAT VIOLATIONS ──────────────────────────────────────────────────
    "OSHA Repeat Violation Policy: If an employer has been cited for the same or "
    "similar violation within the past 5 years, the penalty may be up to $156,259 "
    "per violation — 10x the standard penalty for repeat offenders.",

    "OSHA Willful Violation: A willful violation is one committed with intentional "
    "disregard of, or plain indifference to, OSHA requirements. Penalty: "
    "$9,753 to $156,259 per violation. Criminal prosecution possible.",
]


# Violation type to regulation category mapping
# Used to pre-filter regulations before semantic search
VIOLATION_CATEGORY_MAP = {
    "HELMET VIOLATION":  ["helmet", "head protection", "hardhat", "1926.100", "IS 2925"],
    "MASK VIOLATION":    ["mask", "respiratory", "respirator", "1926.103", "IS 9473"],
    "VEST VIOLATION":    ["vest", "high visibility", "visibility", "1926.201", "IS 15809"],
    "GLOVE VIOLATION":   ["glove", "hand protection", "1926.28", "IS 6994"],
    "GOGGLE VIOLATION":  ["gogg", "eye protection", "1926.102", "IS 5983"],
}