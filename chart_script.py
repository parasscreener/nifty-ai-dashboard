# Create a more vertical and professional AI-based Nifty indicator system flowchart
diagram_code = """
flowchart TD
    %% Data Input Layer
    A["📊 Data Input Layer"]:::dataInput
    A1["🌐 Yahoo Finance API"]:::dataInput
    A2["⚡ Real-time Feeds"]:::dataInput
    A3["📈 Historical OHLC"]:::dataInput
    A4["📰 Volume & News"]:::dataInput
    
    %% Data Preprocessing Layer
    B["🔧 Data Preprocessing"]:::preprocessing
    B1["✅ Data Validation"]:::preprocessing
    B2["🔄 Missing Data Handle"]:::preprocessing
    B3["📏 OHLC Consistency"]:::preprocessing
    B4["🎯 Outlier Detection"]:::preprocessing
    
    %% Feature Engineering Layer
    C["⚙️ Feature Engineering"]:::features
    C1["📊 RSI & MACD"]:::features
    C2["📉 Moving Averages"]:::features
    C3["📊 Bollinger Bands"]:::features
    C4["📈 Volume Analysis"]:::features
    
    %% AI/ML Processing Layer
    D["🤖 AI/ML Processing"]:::aiml
    D1["🧠 LSTM Networks"]:::aiml
    D2["🌳 Random Forest"]:::aiml
    D3["🔍 Pattern Recognition"]:::aiml
    D4["⚖️ SVM Classification"]:::aiml
    
    %% Signal Generation Layer
    E["⚡ Signal Generation"]:::signals
    E1["🎯 Composite Scoring"]:::signals
    E2["📊 Confidence Calc"]:::signals
    E3["⚠️ Risk Assessment"]:::signals
    
    %% Output & Visualization Layer
    F["📱 Output & Alerts"]:::output
    F1["📈 Trading Signals"]:::output
    F2["🖥️ Dashboard Display"]:::output
    F3["🚨 Risk Alerts"]:::output
    
    %% Main flow - vertical connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    %% Layer internal connections - simplified
    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    
    B1 --> C1
    B2 --> C2
    B3 --> C3
    B4 --> C4
    
    C1 --> D1
    C2 --> D2
    C3 --> D3
    C4 --> D4
    
    D1 --> E1
    D2 --> E2
    D3 --> E3
    D4 --> E1
    
    E1 --> F1
    E2 --> F2
    E3 --> F3
    
    %% Class definitions with professional colors
    classDef dataInput fill:#B3E5EC,stroke:#1FB8CD,stroke-width:3px,color:#000,font-weight:bold
    classDef preprocessing fill:#E6F3FF,stroke:#5D878F,stroke-width:3px,color:#000,font-weight:bold
    classDef features fill:#A5D6A7,stroke:#2E8B57,stroke-width:3px,color:#000,font-weight:bold
    classDef aiml fill:#FFCDD2,stroke:#DB4545,stroke-width:3px,color:#000,font-weight:bold
    classDef signals fill:#FFE0B2,stroke:#D2BA4C,stroke-width:3px,color:#000,font-weight:bold
    classDef output fill:#FFEB8A,stroke:#D2BA4C,stroke-width:3px,color:#000,font-weight:bold
"""

# Create the mermaid diagram with better vertical layout and icons
png_path, svg_path = create_mermaid_diagram(
    diagram_code, 
    'nifty_ai_system_flowchart.png',
    'nifty_ai_system_flowchart.svg',
    width=1200,
    height=1400
)

print(f"Professional flowchart saved as PNG: {png_path}")
print(f"Professional flowchart saved as SVG: {svg_path}")