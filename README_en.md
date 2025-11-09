# 🚀 Local LLM Inference Speed Test Tool v1.9 (Chao Modded Version)

![Version](https://img.shields.io/badge/version-1.9-blue.svg)

---

🇬🇧 [**View in English**](#) | 🇨🇳 [**查看中文版 (View in Chinese)**](README.md)

---

## Introduction

This project is a powerful browser-side local large language model (LLM) inference performance testing tool. It aims to help users quickly and conveniently test the Prefill (prompt processing) and Decode (token generation/output) performance of various locally deployed LLM inference services. It runs offline directly in the browser, requiring no server-side deployment, thus ensuring data privacy and testing convenience.

This tool is a modified version based on the original work by "Zhiyuan Suifeng" (Bilibili/DeepSeek-R1-0528), adding several practical features and optimizations. These include a retry mechanism, automatic chart display, separation of Prefill and Decode speeds, fixes for concurrency issues, request timeout control, and automatic configuration saving.

## Changelog

<!-- Please note, the following changelog starts from v1.7. -->

### v1.9 (Chao Enhanced Modded Version - History & Detail View)
*   **📚 Automatic History Saving**: Test results automatically saved to browser localStorage (up to 20 records)
*   **🗂️ History Management Panel**: Beautiful new history modal with easy-to-use interface
    *   View test time, notes, configuration, and performance statistics at a glance
    *   View complete test details (configuration, statistics, full result table)
    *   Export individual history records as CSV
    *   Delete individual records or clear all
*   **📊 History Comparison Feature**: Select multiple history records and generate comparison charts with one click
    *   Quick checkbox selection for multiple test records
    *   Real-time display of selected count
    *   Auto-scroll to comparison chart area
    *   Fully compatible with CSV import feature
*   **🔍 Test Detail View**: Each test result row has a "View Details" button
    *   View complete prompt content and model output
    *   Display detailed request configuration (Temperature, Top P, etc.)
    *   Show comprehensive performance metrics (Prefill/Decode time and speed)
    *   Distinguish reasoning content and completion content (supports reasoning models)
    *   One-click copy prompt or output content
    *   Beautiful modal interface with auto-wrapping text
*   **💾 Data Persistence**: All history records saved in localStorage, no data loss
*   **🎨 Full Bilingual Support**: All new features fully support Chinese/English switching

### v1.8 (Chao Enhanced Modded Version)
*   **🌐 Bilingual Support (Chinese/English)**: Added full bilingual support with a prominent language toggle button. The interface automatically switches between Chinese and English, with the selected language saved to localStorage.
*   **📁 Dual Language File Versions**:
    *   `本地大模型推理速度测试工具v1.9.html` - Chinese version (default Chinese UI)
    *   `LLM_Speed_Test_Tool_v1.9_EN.html` - English version (default English UI)
*   **Export Charts Button**: Added a one-click button to export the current Prefill and Decode charts as a single image.
*   **Import Multiple CSV Results for Comparison**: New feature allowing the import of multiple historical test CSV files for comparative analysis in the charts.
*   **Combined Chart Export**: Export both Prefill and Decode charts as a single image for easy sharing and record-keeping.
*   **Comparison Labels**: Imported CSV file names are now used as legend labels for comparison charts, making it easier to distinguish different test results.
*   **Comparison Export**: Added a one-click export feature for comparison charts, automatically generating filenames that include parameters for better management.
*   **【Important Fix】Support for Reasoning Models**: Now parses the `usage` field to get accurate token counts (`reasoning_tokens` + `completion_tokens`).
*   **Added Token Estimation Fallback**: Uses character-based estimation when the API does not return `usage`.
*   **【Critical Fix】Prefill Speed Calculation Error**: Now uses the actual `prompt_tokens` returned by the API instead of estimated values (fixes ~10% error).
*   **【Compatibility Fix】Support for Multiple Streaming Response Formats**: `delta.reasoning_content` (reasoning)/ `delta.content` / `message.content` / `text`.
*   **Auto-save/Restore All Configuration Parameters**: API address, model, temperature, concurrency, and all other parameters are automatically saved and restored on next open.
*   **【Reasoning Model Fix】Correctly Calculates Total Reasoning+Completion Tokens**: Uses content estimation when `usage` is inaccurate to ensure accurate decode speed.
*   **【Performance Measurement Optimization】Prioritizes Server-Returned Real GPU Processing Time**: (`prompt_eval_duration`/`eval_duration`) to eliminate network latency impact.
*   **【Prefix Cache Fix】Unique Prompts for Concurrent Tests**: Each concurrent request now gets a unique prompt prefix to prevent vLLM's prefix cache from artificially inflating performance metrics.

### v1.7 (Chao Modded Version)

*   Added 3 retries (with 1.5-second intervals) for each request to improve test stability and filter out momentary failures.
*   Automatic chart display, no clicking required, improving usability.
*   Separated Prefill and Decode speeds into two charts for clearer performance visualization.
*   Fixed `Promise.all` meltdown issues under high concurrency, replaced with `Promise.allSettled` to ensure partial request failures do not affect the overall concurrency flow.
*   Added `AbortController` based request timeout.
*   Saved current configuration to `localStorage`, automatically loaded on next open.
*   Optimized UI styling and user experience.


## Features

*   **🌐 Full Bilingual Support**: Complete Chinese/English interface with instant language switching. Language preference is automatically saved.
*   **Browser-Side Local Execution**: Completely offline, data stays local, ensuring privacy and security.
*   **Supports Multiple API Types**:
    *   **OpenAI Compatible API**: Works with all LLM services compatible with the OpenAI API, such as vLLM, TGI, FastChat, etc.
    *   **Ollama API**: Directly utilizes performance metrics provided by Ollama, simplifying testing.
*   **Detailed Performance Metrics**: Measures prompt length, prefill duration, prefill speed, output token count, output duration, and output speed, including P50/P90/P95 percentile statistics.
*   **Concurrent Testing**: Supports setting the number of concurrent requests to simulate high-load scenarios.
*   **Request Retry Mechanism**: Each request includes 3 retries (1.5-second intervals) to effectively filter out transient/occasional network or server failures.
*   **Request Timeout Control**: Implemented with AbortController to prevent long waits for unresponsive requests.
*   **Real-time Progress Bar & Result Display**: Visualizes test progress and updates results in real-time.
*   **Automatic Chart Generation**: Automatically generates and updates Prefill and Decode throughput charts during testing, visually showing performance trends.
*   **📚 History Management** (v1.9 New):
    *   **Auto-save**: Test results automatically saved to localStorage (up to 20 records)
    *   **History Panel**: Beautiful modal interface to view all historical tests
    *   **Detail View**: View complete configuration, statistics, and test results for history records
    *   **Individual Export**: Export history records as CSV files
    *   **Record Management**: Delete individual records or clear all
    *   **Comparison**: Select multiple history records and generate comparison charts with one click
*   **🔍 Test Detail View** (v1.9 New):
    *   **Detail Button**: Each test result row has a "View Details" button
    *   **Full Content**: View complete prompt and model output content
    *   **Request Config**: Display all request parameters (API type, model, Temperature, Top P, etc.)
    *   **Performance Metrics**: Show detailed Prefill/Decode time and speed
    *   **Reasoning Model Support**: Distinguish reasoning content and completion content
    *   **Copy Function**: One-click copy prompt or output content to clipboard
    *   **Optimized Layout**: Auto-wrapping text, optimized table column widths, no horizontal scrolling
*   **Automatic Configuration Saving**: Automatically saves the current API address, model name, and API type to browser `localStorage`, loading them automatically next time.
*   **Result Export**: Supports copying test results as a Markdown table or exporting them as a CSV file.
*   **Chart Export**: Supports one-click export of generated Prefill and Decode throughput charts as an image.
*   **CSV Result Import and Comparison**: Supports importing multiple historical CSV test results and comparing them in the charts.
*   **Flexible Parameter Configuration**: Customizable prompt length range, step, output length, Temperature, Top P, Penalty, etc.

## Core Concepts

*   **OpenAI Compatible API**: Refers to LLM inference APIs that conform to the OpenAI `v1/chat/completions` standard. This tool measures Prefill speed by sending `stream=true` requests and calculating the time until the first token arrives, and Decode speed by calculating the total output completion time.
*   **Ollama API**: Refers to the `api/chat` interface provided by Ollama. This tool sends `stream=false` requests and directly uses metrics like `prompt_eval_duration` and `eval_duration` returned in the Ollama response to calculate performance.
*   **Prefill Throughput**: Measures the speed at which the model processes the input prompt, typically in **tokens/second**. For streaming output, this is calculated from the start of the request until the first token arrives.
*   **Decode Throughput**: Measures the speed at which the model generates output tokens, typically in **tokens/second**. For streaming output, this is calculated from the arrival of the first token until all output is complete.

## How to Use

### 1. Local Run

1.  **Choose your preferred version**:
    *   For English interface by default: Download `LLM_Speed_Test_Tool_v1.9_EN.html`
    *   For Chinese interface by default: Download `本地大模型推理速度测试工具v1.9.html`
2.  Open the HTML file with any modern browser (e.g., Chrome, Firefox, Edge). No additional installation or server is required.
3.  **Language Toggle**: Click the language toggle button (with purple gradient) in the top-right corner to switch between English and Chinese at any time. Your preference is saved automatically.

### 2. Configure Parameters

Fill in or select the following key parameters on the page:

*   **API Type Selection**: Choose `OpenAI Compatible API` or `Ollama API`.
*   **API Address**: The full URL of your LLM inference service API.
    *   OpenAI Compatible API: Typically `http://your_ip:port/v1/chat/completions`
    *   Ollama API: Typically `http://your_ip:port/api/chat`
*   **Model Name**: The name or ID of the model you are using in the API.
*   **API-Key**: If your API requires authentication, fill it in. Leave blank if not needed.
*   **Notes**: Fill in device info, model info, inference framework, etc., for later review.
*   **Minimum Prompt Length**: The starting token count for the prompt in the test.
*   **Maximum Prompt Length**: The ending token count for the prompt in the test.
*   **Step**: The increment in prompt length for each test run.
*   **Expected Output Length**: The maximum number of tokens the model should generate for each request.
*   **Concurrency**: The number of requests sent simultaneously, used for testing concurrent performance.
*   **Request Timeout (ms)**: The maximum waiting time (in milliseconds) for a single request.
*   **Temperature / Top P / Presence Penalty / Frequency Penalty**: Generation parameters used to control the randomness and diversity of model output.

### 3. Start Test

Click the `Start Test` button. The test will send requests sequentially according to your set prompt length range and step.

### 4. Stop Test

Click the `Stop Test` button (displayed after the test starts) to interrupt an ongoing test at any time.

### 5. View Results

Test results will be displayed live in the table at the bottom of the page, including:
*   **Prompt Length (tokens)**
*   **Prefill Duration (ms)**
*   **Prefill Speed (tokens/s)**
*   **Output Length (tokens)**
*   **Output Duration (ms)**
*   **Output Speed (tokens/s)**
*   **Status** (Success/Failure)
*   **Concurrency Statistics**: After the test completes, it will display the minimum, maximum, and average values for overall Prefill and Decode throughput.

Simultaneously, two charts, `Prefill Throughput` and `Decode Throughput`, will automatically generate and update in real-time, visually showing performance trends as prompt length changes.

### 6. Copy Markdown Table

After the test, click the `Copy Markdown Table` button to copy the test data in Markdown table format to your clipboard, convenient for pasting into documents or GitHub READMEs.

### 7. Export CSV Data

After the test, click the `Export CSV Data` button to export the test data as a CSV file, useful for further analysis in Excel or other data analysis tools.
To facilitate version comparison, it's recommended to include the current version number or test time in the filename when exporting CSV.

### 8. Export Charts

After the test, click the `Export Charts` button above the chart area to export both Prefill and Decode throughput charts combined into a single PNG image.

### 9. Import CSV Data for Comparison

Click the `Import CSV File` button, select one or more previously exported CSV files. The tool will parse the data from these files and plot multiple lines on the Prefill and Decode throughput charts, allowing you to compare the performance of different models, configurations, or versions. The imported CSV filenames will be used as legend labels.

### 10. Export Comparison Charts

After importing CSV files and plotting comparison charts, click the `Export Comparison Charts` button above the charts to export the combined Prefill and Decode throughput charts, including all comparison data, as a single PNG image. The filename will automatically include current test parameters and imported filenames for easier management.

### 11. View History Records (v1.9 New)

Click the `📚 View History` button to open the history management panel. History records are automatically saved after each test completion (up to 20 records).

In the history panel, you can:
*   **View List**: Browse summary information of all historical tests (time, notes, model, configuration, performance statistics)
*   **View Details**: Click the `View Details` button to see complete configuration, statistics, and test result table for a record
*   **Export CSV**: Click the `Export CSV` button to export that history record as a CSV file
*   **Delete Record**: Click the `Delete` button to remove unwanted history records
*   **Clear All**: Click the `Clear All` button to remove all history records with one click

### 12. History Comparison (v1.9 New)

In the history panel, you can select multiple history records for comparison:

1.  Check the records you want to compare in the history list (checkboxes on the left)
2.  The top will show "Selected X records" in real-time
3.  Click the `Generate Comparison Charts` button
4.  The history panel automatically closes and the page scrolls to the comparison chart area
5.  View Prefill and Decode performance comparison curves for multiple history records

**Tip**: The history comparison feature is fully compatible with the CSV import feature, allowing you to use both methods for comparative analysis.

### 13. View Test Details (v1.9 New)

In the test results table, each test result row has a `View Details` button. Click this button to open the detail modal and view complete information about the test:

The detail modal contains the following content:
*   **Basic Information**: API type, model name, prompt length, output length, concurrency, test time
*   **Performance Metrics**: Prefill time/speed, output time/speed, test status
*   **Request Configuration**: Temperature, Top P, Max Tokens and all other request parameters
*   **Prompt Content**: Complete prompt text with one-click copy support
*   **Output Content**: Complete model output text with one-click copy support
    *   For reasoning models (e.g., DeepSeek-R1), reasoning content and completion content are displayed separately
    *   All text auto-wraps for easy reading

**Usage Tips**:
*   Click outside the modal or the "Close" button to close the detail modal
*   Click the "Copy Content" button to quickly copy prompt or output content
*   Modal content is scrollable for viewing long texts

## Configuration Details

Here are detailed explanations for each configuration item:

*   **API Address (`apiUrl`)**:
    *   **OpenAI Compatible**: E.g., `http://192.168.1.100:8000/v1/chat/completions`. Ensure your LLM inference framework (e.g., vLLM, TGI) is running in OpenAI API compatible mode.
    *   **Ollama**: E.g., `http://localhost:11434/api/chat`. Ensure the Ollama service is running.
*   **Model Name (`modelName`)**:
    *   **OpenAI Compatible**: The model ID or name used by your service, e.g., `qwen-7b-chat`, `mistral-7b-instruct`.
    *   **Ollama**: The model name you pulled in Ollama, e.g., `llama2`, `qwen`.
*   **API-Key (`apiKey`)**: If your LLM service requires API Key authentication, enter it here. Otherwise, it can be left blank.
*   **Notes (`notes`)**: Used to record environment information for each test, e.g., `GeForce RTX 4090, vLLM 0.3.0, Qwen-7B-Chat-AWQ`.
*   **Minimum/Maximum Prompt Length (`minLength`, `maxLength`)**: Defines the token length range for the test prompts.
*   **Step (`step`)**: The amount by which the prompt length increases in each test. For example, `minLength=128, maxLength=1024, step=128` will test prompts of 128, 256, 384...1024 tokens.
*   **Expected Output Length (`outputLength`)**: The number of tokens the model is expected to generate for each request.
*   **Concurrency (`concurrency`)**: The number of requests sent simultaneously. For example, setting it to `4` means that for each prompt length, `4` requests will be made concurrently, and their average performance and total throughput will be calculated.
*   **Request Timeout (ms) (`timeout`)**: The maximum waiting time (in milliseconds) for a single request. If the model fails to complete the response within this time, the request will be considered failed.
*   **Temperature (`temperatureInput`)**: Controls the randomness of model output. Higher values lead to more random output.
*   **Top P (`topPInput`)**: Controls the range of vocabulary the model selects from high to low probability. For example, `0.9` means the model will select vocabulary with a cumulative probability of up to 90%.
*   **Presence Penalty (`presencePenaltyInput`) / Frequency Penalty (`frequencyPenaltyInput`)**: Used to reduce repetition. Presence Penalty penalizes tokens that have already appeared, while Frequency Penalty penalizes frequently appearing tokens. Ollama API typically combines these into `repeat_penalty`.

## Troubleshooting

*   **Test long-time unresponsive/failure**:
    *   Check if the `API Address` is correct and the network is reachable.
    *   Check if the `Model Name` matches the server configuration.
    *   Check if the `API-Key` is correct.
    *   Increase `Request Timeout (ms)`.
    *   For OpenAI compatible APIs, ensure your service supports streaming (`stream: true`) responses.
*   **Ollama API test fails with missing `duration` field**:
    *   Ensure your Ollama service is a newer version and returns the necessary performance metrics.
*   **Charts not displayed or data anomalous**:
    *   Check the browser console (F12) for error messages.
    *   Clear browser cache and retry.
*   **High concurrency test results vary greatly**:
    *   Ensure your hardware resources (GPU VRAM, CPU threads) can support high concurrency.
    *   Other background loads might interfere with test results.
    *   Model service configurations (e.g., maximum batch size) might affect performance under high concurrency.
*   **CSV import charts not displaying data or showing errors**:
    *   Ensure the imported CSV file is in the standard format exported by this tool.
    *   Check if the column names in the CSV file match the expected names (e.g., "Prompt Length (tokens)", "Prefill Speed (tokens/s)", "Output Speed (tokens/s)").

## Contribution & Feedback

Welcome to submit issues or pull requests to improve this tool or report bugs.

---

**Original Author: 纸鸢随风 (Bilibili/DeepSeek-R1-0528)**
**Based on the latest version by the original author, see:**
**https://www.bilibili.com/opus/1078272739661316119**

**Modded Version Maintainer: chao (QQ Group: 1028429001)**
**Latest Version can be found at: https://github.com/gengchaogit/llm_speedtest**
