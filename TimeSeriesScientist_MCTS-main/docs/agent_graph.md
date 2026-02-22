# TimeSeriesScientist Agent Graph

![Agent Graph 流程图](../assets/agent_graph.png)

## LangGraph 单切片工作流

```mermaid
flowchart LR
    subgraph "Agent Graph (每个 slice 执行一次)"
        START([开始]) --> preprocess
        preprocess[PreprocessAgent<br/>数据清洗 · 可视化] --> analyze
        analyze[AnalysisAgent<br/>趋势 · 季节性 · 平稳性] --> validate
        validate[ValidationAgent<br/>模型选择 · 超参优化] --> forecast
        forecast[ForecastAgent<br/>预测 · 集成] --> report
        report[ReportAgent<br/>生成报告] --> END([结束])
    end
    
    style preprocess fill:#e1f5fe
    style analyze fill:#f3e5f5
    style validate fill:#e8f5e9
    style forecast fill:#fff3e0
    style report fill:#fce4ec
```

## State 数据流

```mermaid
flowchart TB
    subgraph "Input State"
        V[validation_data]
        T[test_data]
        SI[slice_info]
        C[config]
    end
    
    subgraph "Preprocess 输出"
        PD[preprocessed_data]
        PR[preprocess_result]
    end
    
    subgraph "Analyze 输出"
        AR[analysis_result]
    end
    
    subgraph "Validate 输出"
        SM[selected_models]
        BH[best_hyperparameters]
        VR[validation_result]
    end
    
    subgraph "Forecast 输出"
        FR[forecast_result]
    end
    
    subgraph "Report 输出"
        R[report]
    end
    
    V --> PD
    PD --> AR
    AR --> SM
    SM --> FR
    FR --> R
```

## 完整 run() 流程（含切片循环与聚合）

```mermaid
flowchart TB
    subgraph "初始化"
        A1[加载数据] --> A2[转为时序格式]
        A2 --> A3[DataSplitter.create_slices]
    end
    
    subgraph "切片循环"
        B1["Slice 0<br/>invoke(graph)"]
        B2["Slice 1<br/>invoke(graph)"]
        B3["Slice ..."]
        B4["Slice N-1<br/>invoke(graph)"]
        B1 --> B2 --> B3 --> B4
    end
    
    subgraph "每次 invoke 执行"
        C[preprocess → analyze → validate → forecast → report]
    end
    
    subgraph "聚合"
        D[_aggregate_slice_results<br/>平均预测 · 平均指标]
    end
    
    A3 --> B1
    B4 --> D
    B1 -.-> C
    B2 -.-> C
    B4 -.-> C
    
    D --> E[返回 all_results + aggregated_results]
```

## 节点职责速查

| 节点 | Agent | 输入 | 输出 |
|------|-------|------|------|
| preprocess | PreprocessAgent | validation_data | preprocessed_data, preprocess_result |
| analyze | AnalysisAgent | preprocessed_data, visualizations | analysis_result |
| validate | ValidationAgent | analysis_result, available_models, preprocessed_data | validation_result, selected_models, best_hyperparameters |
| forecast | ForecastAgent | selected_models, best_hyperparameters, test_data | forecast_result |
| report | ReportAgent | experiment_summary | report |
