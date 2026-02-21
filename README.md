# 🤖 AI-Driven Database Insight Generator

An enterprise-grade analytical engine that transforms static `.db` files into dynamic, AI-powered intelligence reports. This application automates schema discovery, statistical profiling, and anomaly detection across multiple database environments.



## 🚀 Core Capabilities

### 🔍 Single-File Intelligence
- **Automated Schema Mapping:** Instant visualization of tables and relationships.
- **AI Anomaly Detection:** Utilizes **Isolation Forest** algorithms to identify outliers and data corruption without manual thresholds.
- **Statistical Profiling:** Real-time generation of skewness, kurtosis, and distribution metrics.
- **Correlation Heatmaps:** Visual mapping of feature dependencies using Pearson/Spearman coefficients.

### 🌐 Cross-DB Synthesis
- **Relational Discovery:** Identifies potential `JOIN` keys and shared schemas across disparate files.
- **Conflict Analysis:** Detects data inconsistencies and record duplicates between databases.
- **Trend Synchronization:** Correlates time-series data across multiple sources to identify macro patterns.

## 🛠 Tech Stack
- **Frontend/UI:** [Dash by Plotly](https://dash.plotly.com/) (Reactive Analytics)
- **Data Engine:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (Isolation Forest)
- **Database Support:** SQLite, PostgreSQL
- **Deployment:** Render (PaaS)



## 📥 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/ai-db-insight-generator.git](https://github.com/your-username/ai-db-insight-generator.git)
   cd ai-db-insight-generator
