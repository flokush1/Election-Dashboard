# Electoral Analytics Dashboard - System Architecture

## Comprehensive Hierarchical Flowchart

```mermaid
graph TB
    subgraph Header["🏛️ ELECTORAL ANALYTICS DASHBOARD - Multi-Level Voter Intelligence Platform"]
    end

    subgraph Level1["LEVEL 1: PARLIAMENT CONSTITUENCY 🏛️"]
        P1[📊 Aggregate Metrics<br/>Total Votes | Population<br/>Assemblies | Booths | Turnout]
        P2[🎯 Party Performance<br/>BJP | Congress | AAP<br/>Seats Won | Margins]
        P3[👥 Demographics<br/>Age: 18-25, 26-35, 36-45, 46-60, 60+<br/>Gender: M/F/O Ratio<br/>Religion: Hindu, Muslim, Sikh, Christian, Jain, Buddhist<br/>Caste: Brahmin, Kshatriya, Vaishya, OBC, SC, ST]
        P4[💰 Economic Indicators<br/>Categories A-G<br/>Land Rate | Construction Cost]
        P5[🗺️ Interactive Map<br/>Assembly Boundaries]
    end

    subgraph Level2["LEVEL 2: ASSEMBLY CONSTITUENCY 🗺️"]
        A1[📊 Assembly Metrics<br/>Name & Number | Total Votes<br/>Population | Wards | Booths | Turnout]
        A2[🎯 Party Results<br/>BJP/Congress/AAP Votes & %<br/>Booths Won | Margins | Winner]
        A3[👥 Demographics Breakdown<br/>Age Groups with Counts<br/>Gender Distribution & Ratios<br/>Religion Composition %<br/>Caste Breakdown %]
        A4[💰 Economic Profile<br/>Categories Distribution<br/>Land & Construction Rates]
        A5[🗺️ Ward-Level Map]
    end

    subgraph Level3["LEVEL 3: MUNICIPAL WARD 🏘️"]
        W1[📊 Ward Metrics<br/>Name & Number | Votes<br/>Population | Booths | Turnout]
        W2[🎯 Party Results<br/>BJP/Congress/AAP votes & %<br/>Booths Won | Margin | Leader]
        W3[👥 Demographics<br/>Age Groups | Gender<br/>Religion | Caste]
        W4[💰 Economic Data<br/>Dominant Category<br/>Land Rate Range]
        W5[🗺️ Booth-Level Map]
    end

    subgraph Level4["LEVEL 4: POLLING BOOTH 🗳️"]
        B1[📊 Booth Details<br/>Booth Number | Address<br/>Coordinates | Registered Voters<br/>Votes Polled | Turnout %]
        B2[🎯 Exact Party Results<br/>BJP: votes + %<br/>Congress: votes + %<br/>AAP: votes + %<br/>Others: votes + %<br/>NOTA: votes + %<br/>Winner & Margin]
        B3[👥 Precise Demographics<br/>Age: Exact numbers + ratios<br/>Gender: Exact numbers + ratios<br/>Religion: Exact breakdown<br/>Caste: Exact breakdown]
        B4[💰 Economic Indicators<br/>Category Code A-G<br/>Land Rate per sqm<br/>Construction Cost<br/>Locality Name]
        B5[🗺️ Pinpoint Location]
    end

    subgraph Level5["LEVEL 5: HOUSEHOLD/BUILDING 🏠"]
        H1[🏠 Building Info<br/>Building Name/ID<br/>Street Address | House Number<br/>Coordinates | Building Type]
        H2[👥 Household Demographics<br/>Total Members | Registered Voters<br/>Age Distribution | Gender<br/>Primary Religion | Caste<br/>Family Head]
        H3[👤 Voter Details<br/>Voter IDs | Names<br/>Age | Gender | Serial Numbers<br/>Relationship to Head]
        H4[💰 Economic Profile<br/>Household Category]
        H5[📊 Voting History<br/>Previous Turnout<br/>Family Pattern<br/>Historical Preference]
        H6[🎯 ML Predictions<br/>Family Party Preference<br/>Household Turnout Probability<br/>Individual Predictions<br/>Confidence Score<br/>Family Influencer]
    end

    subgraph AIPanel["🤖 AI VOTER PREDICTION PANEL"]
        subgraph Section1["SECTION 1: DATA INPUT & UPLOAD ☁️"]
            AI1[📤 Upload ML Model<br/>.pkl / .pth files<br/>scikit-learn | PyTorch<br/>Pre-trained: Madipur, Moti Nagar, RK Puram<br/>Status: Connected/Disconnected]
            AI2[📊 Upload Voter Data<br/>Excel/CSV Upload<br/>Auto-detect Columns<br/>Schema Mapping<br/>Preview | Total Loaded]
        end

        subgraph Section2["SECTION 2: ML FEATURE PROCESSING ⚙️"]
            AI3[📋 Demographic Features<br/>Voter ID | Name | Age Groups<br/>Gender | Religion | Caste]
            AI4[📍 Location Features<br/>Assembly | Ward | Booth<br/>Locality | Coordinates]
            AI5[💰 Economic Features<br/>Category A-G | Land Rate<br/>Construction Cost | Income Proxy]
            AI6[📈 Behavioral Features<br/>Turnout History | Previous Elections<br/>Booth Trends | Neighborhood Patterns]
            AI7[👨‍👩‍👧‍👦 Family Context<br/>Household Members<br/>Family History | Influencer]
            
            AI8[🧠 ML Model Processing<br/>Algorithm: Random Forest/Neural Net<br/>Training Status<br/>Feature Importance<br/>Confidence Metrics]
        end

        subgraph Section3["SECTION 3: PREDICTION OUTPUTS 🎯"]
            AI9[👤 Individual Voter Prediction<br/>━━━━━━━━━━━━━━━━━<br/>Voter Info Display<br/>━━━━━━━━━━━━━━━━━<br/>🔶 BJP: ████████ 45%<br/>🔷 Congress: ██ 12%<br/>🔵 AAP: ████████████ 38%<br/>⚪ Others: █ 4%<br/>⚫ NOTA: 1%<br/>━━━━━━━━━━━━━━━━━<br/>✅ PREDICTED: BJP<br/>━━━━━━━━━━━━━━━━━<br/>📊 Turnout: 78% HIGH<br/>⭐ Confidence: HIGH 92%]
            
            AI10[👨‍👩‍👧‍👦 Family Analysis<br/>━━━━━━━━━━━━━━━━━<br/>Total Members: 5<br/>Registered Voters: 4<br/>━━━━━━━━━━━━━━━━━<br/>Family Preference: BJP<br/>Split Indicator: Unified<br/>Influencer: Father<br/>Influencer Confidence: 88%<br/>━━━━━━━━━━━━━━━━━<br/>Expected Turnout: 75%<br/>Alignment: 90%<br/>Strategic Importance: HIGH]
            
            AI11[📥 Download & Export<br/>Export CSV | Full Report<br/>━━━━━━━━━━━━━━━━━<br/>Data Preview Table<br/>Filter | Sort | Pagination<br/>━━━━━━━━━━━━━━━━━<br/>📄 All Predictions CSV<br/>📄 Filtered Data CSV<br/>📄 High-Confidence CSV<br/>📄 Summary Report PDF]
        end
    end

    subgraph Charts["📊 VISUALIZATION COMPONENTS"]
        C1[📊 Age Distribution<br/>Bar Chart]
        C2[🥧 Gender Composition<br/>Pie Chart]
        C3[📊 Religion Distribution<br/>Multi-Bar Chart]
        C4[📊 Caste Composition<br/>Bar Chart]
        C5[📊 Party Performance<br/>Bar Chart]
        C6[📊 Economic Categories<br/>Distribution A-G]
        C7[📊 Predicted vs Actual<br/>Comparison Chart]
    end

    subgraph Legend["LEGEND"]
        L1[🔵 Data Aggregation ↑]
        L2[🟢 User Drill-Down ↓]
        L3[🟠 Voter Data → ML]
        L4[🟣 Predictions → Dashboard]
        L5[🔴 Model Improvement Loop]
        L6[━━━━━━━━━━━━━━━]
        L7[🔶 BJP Orange #FF9933]
        L8[🔷 Congress Blue #3B82F6]
        L9[🔵 AAP Cyan #06B6D4]
        L10[━━━━━━━━━━━━━━━]
        L11[🟢 High Confidence]
        L12[🟡 Medium Confidence]
        L13[🔴 Low Confidence]
    end

    %% Hierarchical Navigation Flow (Drill-Down)
    Level1 -->|Drill Down| Level2
    Level2 -->|Drill Down| Level3
    Level3 -->|Drill Down| Level4
    Level4 -->|Drill Down| Level5

    %% Data Aggregation Flow (Bottom-Up)
    Level5 -.->|Aggregate| Level4
    Level4 -.->|Aggregate| Level3
    Level3 -.->|Aggregate| Level2
    Level2 -.->|Aggregate| Level1

    %% AI Integration Flows
    Level5 -->|Voter Data| AIPanel
    Level4 -->|Voter Data| AIPanel
    Level3 -->|Voter Data| AIPanel
    Level2 -->|Voter Data| AIPanel
    Level1 -->|Voter Data| AIPanel

    AIPanel -->|Predictions| Level5
    AIPanel -->|Predictions| Level4
    AIPanel -->|Predictions| Level3
    AIPanel -->|Predictions| Level2
    AIPanel -->|Predictions| Level1

    %% AI Internal Flow
    AI1 --> AI8
    AI2 --> AI8
    AI3 --> AI8
    AI4 --> AI8
    AI5 --> AI8
    AI6 --> AI8
    AI7 --> AI8
    AI8 --> AI9
    AI8 --> AI10
    AI9 --> AI11
    AI10 --> AI11

    %% Charts Connection
    Level1 --> Charts
    Level2 --> Charts
    Level3 --> Charts
    Level4 --> Charts
    Level5 --> Charts
    AIPanel --> Charts

    %% Styling
    classDef level1Style fill:#7C3AED,stroke:#5B21B6,stroke-width:3px,color:#fff
    classDef level2Style fill:#3B82F6,stroke:#1D4ED8,stroke-width:3px,color:#fff
    classDef level3Style fill:#10B981,stroke:#059669,stroke-width:3px,color:#fff
    classDef level4Style fill:#F97316,stroke:#C2410C,stroke-width:3px,color:#fff
    classDef level5Style fill:#DC2626,stroke:#991B1B,stroke-width:3px,color:#fff
    classDef aiStyle fill:#FCD34D,stroke:#F59E0B,stroke-width:3px,color:#000
    classDef chartStyle fill:#8B5CF6,stroke:#6D28D9,stroke-width:2px,color:#fff
    classDef legendStyle fill:#E5E7EB,stroke:#9CA3AF,stroke-width:2px,color:#000

    class P1,P2,P3,P4,P5 level1Style
    class A1,A2,A3,A4,A5 level2Style
    class W1,W2,W3,W4,W5 level3Style
    class B1,B2,B3,B4,B5 level4Style
    class H1,H2,H3,H4,H5,H6 level5Style
    class AI1,AI2,AI3,AI4,AI5,AI6,AI7,AI8,AI9,AI10,AI11 aiStyle
    class C1,C2,C3,C4,C5,C6,C7 chartStyle
    class L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13 legendStyle
```

## System Overview

### Hierarchical Structure
The dashboard implements a **5-tier hierarchical navigation system**:

1. **Parliament Level** 🏛️ - Constituency-wide overview
2. **Assembly Level** 🗺️ - Regional breakdown
3. **Ward Level** 🏘️ - Municipal unit analysis
4. **Booth Level** 🗳️ - Ground-level granularity
5. **Household Level** 🏠 - Building/family-specific data

### Key Components

#### Demographics Analysis (All Levels)
- **Age Groups**: 18-25, 26-35, 36-45, 46-60, 60+
- **Gender**: Male, Female, Others (with M/F ratio)
- **Religion**: Hindu, Muslim, Sikh, Christian, Jain, Buddhist, Unknown
- **Caste**: Brahmin, Kshatriya, Vaishya, OBC, SC, ST, No Caste System, Unknown

#### Economic Indicators
- **Categories**: A (Elite) through G (Low Income)
- **Land Rates**: Per square meter
- **Construction Costs**: Per square meter

#### Party Performance
- **BJP** 🔶 (Orange)
- **Congress** 🔷 (Blue)
- **AAP** 🔵 (Cyan)
- **Others** ⚪ (Gray)
- **NOTA** ⚫ (Black)

### AI Prediction Engine

#### Input Pipeline
1. Upload ML model (.pkl/.pth)
2. Upload voter Excel/CSV data
3. Auto-detect and map columns
4. Feature engineering across 5 dimensions:
   - Demographics
   - Location
   - Economic
   - Behavioral
   - Family Context

#### Processing
- Random Forest / Neural Network algorithms
- Feature importance scoring
- Confidence metric calculation

#### Outputs
- **Individual Predictions**: Party preference probabilities
- **Turnout Prediction**: Voting likelihood
- **Confidence Scores**: High/Medium/Low
- **Family Analysis**: Household-level patterns
- **Batch Exports**: CSV/PDF reports

### Data Flows

1. **Bottom-Up Aggregation** 🔵: Household → Booth → Ward → Assembly → Parliament
2. **Top-Down Navigation** 🟢: Parliament → Assembly → Ward → Booth → Household
3. **ML Integration** 🟠: All levels feed voter data to AI engine
4. **Prediction Distribution** 🟣: AI predictions flow back to all levels
5. **Continuous Improvement** 🔴: Results feedback loop for model refinement

### Visualization Suite

1. Age Distribution Bar Chart
2. Gender Composition Pie Chart
3. Religion Distribution Chart
4. Caste Composition Chart
5. Party Performance Bar Chart
6. Economic Category Distribution
7. Predicted vs Actual Comparison

---

**Built with**: React 18 • Vite • Tailwind CSS • Leaflet Maps • Recharts • Flask ML API • scikit-learn • PyTorch
