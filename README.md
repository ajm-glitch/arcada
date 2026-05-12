# aRCADA
### A Regional Cabled Array Data Assistant
---

## Overview

The NSF-funded Ocean Observatories Initiative (OOI) [Regional Cabled Array (RCA)](https://oceanobservatories.org/regional-cabled-array/) has generated a decade of continuous observational data spanning geophysics, geochemistry, biology, and oceanography from seafloor instruments deployed off the Oregon coast. This record documents some of the most dynamic environments on Earth — methane seeps, hydrothermal vents, and an active submarine volcano — all unfolding across an entire tectonic plate.

Yet accessing this data reliably remains a significant barrier to research. While some instruments offer clean programmatic access through the OOI API, many PI-operated instruments — including scanning sonar and seafloor mass spectrometers — store unprocessed sensor readings on standalone HTML pages with no metadata or consistent structure. The result: researchers spend more time locating and contextualizing data than analyzing it, and cross-disciplinary integration of data streams is rare.

**aRCADA** is an agentic, AI-powered framework that changes this. It enables researchers to access OOI RCA data through plain-language requests — no API knowledge required.

---

## Motivation

The Cascadia margin is one of the most scientifically compelling regions on the planet. The processes unfolding there — from microbial methane oxidation at cold seeps to tectonic deformation along the Cascadia Subduction Zone — are inherently interdisciplinary. They cannot be understood through any single data stream or field of expertise.

The RCA exists precisely to make this kind of integrative science possible. But fragmented data access undermines that potential. A geochemist who wants to compare methane flux anomalies with seismic activity shouldn't have to become an expert in two separate data retrieval systems. aRCADA closes that gap.

---

## What aRCADA Does

Given a plain-language query like:

> *"Show me bottom pressure and seismic data near Axial Seamount for the two weeks following the 2024 eruption."*

aRCADA will:

1. **Interpret** the request and identify the relevant instruments and data streams
2. **Retrieve** metadata from its knowledge base of OOI API streams and PI-operated instruments
3. **Plan** a multi-step data retrieval strategy using an agentic workflow
4. **Fetch** data from the appropriate sources — whether through the OOI REST API or by parsing PI instrument HTML pages
5. **Return** normalized, formatted data with full provenance metadata


---

*aRCADA is an academic research project developed at the University of Washington.*