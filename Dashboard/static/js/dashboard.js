// document.addEventListener("DOMContentLoaded", () => {
//     // Initialize navigation
//     initNavigation()
  
//     // Load dashboard data
//     loadDashboardStats()
  
//     // Load agents data
//     loadAgents()
  
//     // Initialize prediction form
//     initPredictionForm()
  
//     // Initialize agent selector
//     initAgentSelector()
//   })
  
//   // Navigation between sections
//   function initNavigation() {
//     const navLinks = document.querySelectorAll(".sidebar-menu .nav-link")
  
//     navLinks.forEach((link) => {
//       link.addEventListener("click", function (e) {
//         e.preventDefault()
  
//         // Remove active class from all links
//         navLinks.forEach((l) => l.classList.remove("active"))
  
//         // Add active class to clicked link
//         this.classList.add("active")
  
//         // Hide all content sections
//         document.querySelectorAll(".content-section").forEach((section) => {
//           section.classList.remove("active")
//         })
  
//         // Show the selected content section
//         const sectionId = this.getAttribute("data-section")
//         document.getElementById(sectionId).classList.add("active")
  
//         // Refresh charts if switching to factors tab
//         if (sectionId === "factors") {
//           // Reload feature importance data
//           fetch("/api/feature-importance")
//             .then((response) => response.json())
//             .then((data) => {
//               createFeatureImportanceDetailedChart(data)
//             })
//             .catch((error) => console.error("Error loading feature importance:", error))
  
//           // Reload metrics data
//           fetch("/api/dashboard-stats")
//             .then((response) => response.json())
//             .then((data) => {
//               createMetricsByPerformanceChart(data.avg_metrics_by_performance)
//             })
//             .catch((error) => console.error("Error loading dashboard stats:", error))
//         }
//       })
//     })
//   }
  
//   // Load dashboard statistics
//   function loadDashboardStats() {
//     fetch("/api/dashboard-stats")
//       .then((response) => response.json())
//       .then((data) => {
//         // Update total agents
//         document.getElementById("total-agents").textContent = data.total_agents
  
//         // Update risk distribution
//         const highRisk = data.risk_distribution["High Risk"] || 0
//         const mediumHighRisk = data.risk_distribution["Medium-High Risk"] || 0
//         const mediumLowRisk = data.risk_distribution["Medium-Low Risk"] || 0
//         const lowRisk = data.risk_distribution["Low Risk"] || 0
  
//         document.getElementById("high-risk-agents").textContent = highRisk
//         document.getElementById("medium-risk-agents").textContent = mediumHighRisk + mediumLowRisk
//         document.getElementById("low-risk-agents").textContent = lowRisk
  
//         // Create risk distribution chart
//         createRiskDistributionChart(data.risk_distribution)
  
//         // Create performance distribution chart
//         createPerformanceDistributionChart(data.performance_distribution)
  
//         // Create metrics by performance chart - ensure this runs
//         setTimeout(() => {
//           createMetricsByPerformanceChart(data.avg_metrics_by_performance)
//         }, 500)
//       })
//       .catch((error) => console.error("Error loading dashboard stats:", error))
  
//     // Load feature importance
//     fetch("/api/feature-importance")
//       .then((response) => response.json())
//       .then((data) => {
//         // Create feature importance charts
//         createFeatureImportanceChart(data)
  
//         // Ensure this runs with a slight delay to make sure the DOM is ready
//         setTimeout(() => {
//           createFeatureImportanceDetailedChart(data)
//         }, 500)
//       })
//       .catch((error) => console.error("Error loading feature importance:", error))
//   }
  
//   // Load agents data
//   function loadAgents() {
//     fetch("/api/agents")
//       .then((response) => response.json())
//       .then((data) => {
//         // Populate agents table
//         populateAgentsTable(data)
  
//         // Populate agent selector
//         populateAgentSelector(data)
  
//         // Create prediction distribution chart
//         createPredictionDistributionChart(data)
//       })
//       .catch((error) => console.error("Error loading agents:", error))
//   }
  
//   // Populate agents table
//   function populateAgentsTable(agents) {
//     const tableBody = document.querySelector("#agents-table tbody")
//     tableBody.innerHTML = ""
  
//     agents.forEach((agent) => {
//       const row = document.createElement("tr")
  
//       // Create risk badge class
//       let riskBadgeClass = ""
//       switch (agent.risk_segment) {
//         case "High Risk":
//           riskBadgeClass = "badge-high-risk"
//           break
//         case "Medium-High Risk":
//           riskBadgeClass = "badge-medium-high-risk"
//           break
//         case "Medium-Low Risk":
//           riskBadgeClass = "badge-medium-low-risk"
//           break
//         case "Low Risk":
//           riskBadgeClass = "badge-low-risk"
//           break
//       }
  
//       // Create performance badge class
//       let performanceBadgeClass = ""
//       switch (agent.performance_class) {
//         case "High":
//           performanceBadgeClass = "badge-high"
//           break
//         case "Medium-High":
//           performanceBadgeClass = "badge-medium-high"
//           break
//         case "Medium-Low":
//           performanceBadgeClass = "badge-medium-low"
//           break
//         case "Low":
//           performanceBadgeClass = "badge-low"
//           break
//       }
  
//       row.innerHTML = `
//               <td>${agent.agent_code}</td>
//               <td>${agent.agent_name}</td>
//               <td>${agent.agent_age}</td>
//               <td>${agent.tenure_months}</td>
//               <td><span class="badge ${performanceBadgeClass}">${agent.performance_class}</span></td>
//               <td><span class="badge ${riskBadgeClass}">${agent.risk_segment}</span></td>
//               <td>${(agent.nill_probability * 100).toFixed(2)}%</td>
//               <td>
//                   <button class="btn btn-sm btn-primary view-agent-btn" data-agent-code="${agent.agent_code}">
//                       View
//                   </button>
//               </td>
//           `
  
//       tableBody.appendChild(row)
//     })
  
//     // Add event listeners to view buttons
//     document.querySelectorAll(".view-agent-btn").forEach((button) => {
//       button.addEventListener("click", function () {
//         const agentCode = this.getAttribute("data-agent-code")
//         openAgentDetailModal(agentCode)
//       })
//     })
//   }
  
//   // Populate agent selector
//   function populateAgentSelector(agents) {
//     const selector = document.getElementById("agent-select")
  
//     // Clear existing options except the first one
//     while (selector.options.length > 1) {
//       selector.remove(1)
//     }
  
//     // Add agents to selector
//     agents.forEach((agent) => {
//       const option = document.createElement("option")
//       option.value = agent.agent_code
//       option.textContent = `${agent.agent_code} - ${agent.agent_name} (${agent.performance_class})`
//       selector.appendChild(option)
//     })
//   }
  
//   // Initialize prediction form
//   function initPredictionForm() {
//     const form = document.getElementById("prediction-form")
  
//     form.addEventListener("submit", (e) => {
//       e.preventDefault()
  
//       // Get form data
//       const formData = {
//         agent_age: Number.parseInt(document.getElementById("agent_age").value),
//         tenure_months: Number.parseInt(document.getElementById("tenure_months").value),
//         unique_proposal: Number.parseInt(document.getElementById("unique_proposal").value),
//         unique_quotations: Number.parseInt(document.getElementById("unique_quotations").value),
//         unique_customers: Number.parseInt(document.getElementById("unique_customers").value),
//         proposal_to_quotation_ratio: Number.parseFloat(document.getElementById("proposal_to_quotation_ratio").value),
//       }
  
//       // Calculate derived features
//       formData.proposal_intensity = formData.unique_proposal / Math.max(1, formData.tenure_months)
//       formData.quotation_intensity = formData.unique_quotations / Math.max(1, formData.tenure_months)
//       formData.customer_intensity = formData.unique_customers / Math.max(1, formData.tenure_months)
  
//       // Make prediction
//       fetch("/api/predict", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify(formData),
//       })
//         .then((response) => response.json())
//         .then((data) => {
//           if (data.error) {
//             alert("Error: " + data.error)
//             return
//           }
  
//           // Show prediction result
//           document.getElementById("prediction-result").style.display = "block"
//           document.getElementById("prediction-probability").textContent = (data.nill_probability * 100).toFixed(2) + "%"
//           document.getElementById("prediction-risk").textContent = data.risk_segment
//           document.getElementById("prediction-performance").textContent = data.performance_class
  
//           // Set alert class based on risk
//           const alertElement = document.getElementById("prediction-alert")
//           alertElement.className = "alert"
  
//           switch (data.risk_segment) {
//             case "High Risk":
//               alertElement.classList.add("alert-danger")
//               break
//             case "Medium-High Risk":
//               alertElement.classList.add("alert-warning")
//               break
//             case "Medium-Low Risk":
//               alertElement.classList.add("alert-info")
//               break
//             case "Low Risk":
//               alertElement.classList.add("alert-success")
//               break
//           }
  
//           // Show recommendations
//           const actionsList = document.getElementById("prediction-actions")
//           actionsList.innerHTML = ""
  
//           data.recommendations.forEach((recommendation) => {
//             const li = document.createElement("li")
//             li.textContent = recommendation
//             actionsList.appendChild(li)
//           })
//         })
//         .catch((error) => console.error("Error making prediction:", error))
//     })
//   }
  
//   // Initialize agent selector
//   function initAgentSelector() {
//     const selector = document.getElementById("agent-select")
  
//     selector.addEventListener("change", function () {
//       const agentCode = this.value
  
//       if (!agentCode) {
//         // Hide agent details if no agent is selected
//         document.getElementById("agent-details").style.display = "none"
//         document.getElementById("agent-factors").style.display = "none"
//         return
//       }
  
//       // Load agent details
//       fetch(`/api/agent/${agentCode}`)
//         .then((response) => response.json())
//         .then((data) => {
//           // Show agent details
//           document.getElementById("agent-details").style.display = "flex"
//           document.getElementById("agent-factors").style.display = "block"
  
//           // Update agent profile
//           document.getElementById("detail-agent-code").textContent = data.agent_code
//           document.getElementById("detail-agent-name").textContent = data.agent_name
//           document.getElementById("detail-agent-age").textContent = data.agent_age
//           document.getElementById("detail-agent-tenure").textContent = data.tenure_months
//           document.getElementById("detail-performance-class").textContent = data.performance_class
//           document.getElementById("detail-risk-level").textContent = data.risk_segment
//           document.getElementById("detail-nill-probability").textContent = (data.nill_probability * 100).toFixed(2) + "%"
  
//           // Update recommendations
//           const recommendationList = document.getElementById("recommendation-list")
//           recommendationList.innerHTML = ""
  
//           data.recommendations.forEach((recommendation) => {
//             const li = document.createElement("li")
//             li.textContent = recommendation
//             recommendationList.appendChild(li)
//           })
  
//           // Update factors
//           const factorsList = document.getElementById("agent-factors-list")
//           factorsList.innerHTML = ""
  
//           data.top_factors.forEach((factor) => {
//             const factorCard = document.createElement("div")
//             factorCard.className = `factor-card ${factor.impact}-impact`
  
//             const factorTitle = document.createElement("div")
//             factorTitle.className = "factor-title"
//             factorTitle.textContent = factor.factor
  
//             const factorDescription = document.createElement("div")
//             factorDescription.className = "factor-description"
//             factorDescription.textContent = factor.description
  
//             const factorImpact = document.createElement("span")
//             factorImpact.className = `badge badge-impact-${factor.impact}`
//             factorImpact.textContent = factor.impact.charAt(0).toUpperCase() + factor.impact.slice(1) + " Impact"
  
//             factorCard.appendChild(factorTitle)
//             factorCard.appendChild(factorDescription)
//             factorCard.appendChild(document.createElement("br"))
//             factorCard.appendChild(factorImpact)
  
//             factorsList.appendChild(factorCard)
//           })
//         })
//         .catch((error) => console.error("Error loading agent details:", error))
//     })
//   }
  
//   // Open agent detail modal
//   function openAgentDetailModal(agentCode) {
//     fetch(`/api/agent/${agentCode}`)
//       .then((response) => response.json())
//       .then((data) => {
//         // Update modal title
//         document.getElementById("agentDetailModalLabel").textContent =
//           `Agent Details: ${data.agent_name} (${data.agent_code})`
  
//         // Update agent profile
//         document.getElementById("modal-agent-code").textContent = data.agent_code
//         document.getElementById("modal-agent-name").textContent = data.agent_name
//         document.getElementById("modal-agent-age").textContent = data.agent_age
//         document.getElementById("modal-agent-tenure").textContent = data.tenure_months
//         document.getElementById("modal-performance-class").textContent = data.performance_class
//         document.getElementById("modal-risk-level").textContent = data.risk_segment
//         document.getElementById("modal-nill-probability").textContent = (data.nill_probability * 100).toFixed(2) + "%"
  
//         // Update agent metrics
//         document.getElementById("modal-unique-proposals").textContent = data.unique_proposal
//         document.getElementById("modal-unique-quotations").textContent = data.unique_quotations
//         document.getElementById("modal-unique-customers").textContent = data.unique_customers
//         document.getElementById("modal-proposal-intensity").textContent = data.proposal_intensity
//           ? data.proposal_intensity.toFixed(2)
//           : "N/A"
//         document.getElementById("modal-quotation-intensity").textContent = data.quotation_intensity
//           ? data.quotation_intensity.toFixed(2)
//           : "N/A"
//         document.getElementById("modal-customer-intensity").textContent = data.customer_intensity
//           ? data.customer_intensity.toFixed(2)
//           : "N/A"
//         document.getElementById("modal-proposal-quotation-ratio").textContent = data.proposal_to_quotation_ratio
//           ? data.proposal_to_quotation_ratio.toFixed(2)
//           : "N/A"
  
//         // Update recommendations
//         const recommendationList = document.getElementById("modal-recommendation-list")
//         recommendationList.innerHTML = ""
  
//         data.recommendations.forEach((recommendation) => {
//           const li = document.createElement("li")
//           li.textContent = recommendation
//           recommendationList.appendChild(li)
//         })
  
//         // Update factors
//         const factorsList = document.getElementById("modal-agent-factors")
//         factorsList.innerHTML = ""
  
//         data.top_factors.forEach((factor) => {
//           const factorCard = document.createElement("div")
//           factorCard.className = `factor-card ${factor.impact}-impact`
  
//           const factorTitle = document.createElement("div")
//           factorTitle.className = "factor-title"
//           factorTitle.textContent = factor.factor
  
//           const factorDescription = document.createElement("div")
//           factorDescription.className = "factor-description"
//           factorDescription.textContent = factor.description
  
//           const factorImpact = document.createElement("span")
//           factorImpact.className = `badge badge-impact-${factor.impact}`
//           factorImpact.textContent = factor.impact.charAt(0).toUpperCase() + factor.impact.slice(1) + " Impact"
  
//           factorCard.appendChild(factorTitle)
//           factorCard.appendChild(factorDescription)
//           factorCard.appendChild(document.createElement("br"))
//           factorCard.appendChild(factorImpact)
  
//           factorsList.appendChild(factorCard)
//         })
  
//         // Show the modal
//         const modal = new bootstrap.Modal(document.getElementById("agentDetailModal"))
//         modal.show()
//       })
//       .catch((error) => console.error("Error loading agent details:", error))
//   }
  
//   // Create risk distribution chart
//   function createRiskDistributionChart(riskDistribution) {
//     const chartElement = document.getElementById("risk-distribution-chart")
//     const chart = echarts.init(chartElement)
  
//     const data = [
//       { value: riskDistribution["High Risk"] || 0, name: "High Risk" },
//       { value: riskDistribution["Medium-High Risk"] || 0, name: "Medium-High Risk" },
//       { value: riskDistribution["Medium-Low Risk"] || 0, name: "Medium-Low Risk" },
//       { value: riskDistribution["Low Risk"] || 0, name: "Low Risk" },
//     ]
  
//     const option = {
//       tooltip: {
//         trigger: "item",
//         formatter: "{a} <br/>{b}: {c} ({d}%)",
//       },
//       legend: {
//         orient: "vertical",
//         right: 10,
//         top: "center",
//         data: ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"],
//       },
//       series: [
//         {
//           name: "Risk Distribution",
//           type: "pie",
//           radius: ["50%", "70%"],
//           avoidLabelOverlap: false,
//           itemStyle: {
//             borderRadius: 10,
//             borderColor: "#fff",
//             borderWidth: 2,
//           },
//           label: {
//             show: false,
//             position: "center",
//           },
//           emphasis: {
//             label: {
//               show: true,
//               fontSize: "18",
//               fontWeight: "bold",
//             },
//           },
//           labelLine: {
//             show: false,
//           },
//           data: data,
//           color: ["#dc3545", "#fd7e14", "#ffc107", "#28a745"],
//         },
//       ],
//     }
  
//     chart.setOption(option)
  
//     // Resize chart on window resize
//     window.addEventListener("resize", () => {
//       chart.resize()
//     })
//   }
  
//   // Create performance distribution chart
//   function createPerformanceDistributionChart(performanceDistribution) {
//     const chartElement = document.getElementById("performance-distribution-chart")
//     const chart = echarts.init(chartElement)
  
//     const data = [
//       { value: performanceDistribution["High"] || 0, name: "High" },
//       { value: performanceDistribution["Medium-High"] || 0, name: "Medium-High" },
//       { value: performanceDistribution["Medium-Low"] || 0, name: "Medium-Low" },
//       { value: performanceDistribution["Low"] || 0, name: "Low" },
//     ]
  
//     const option = {
//       tooltip: {
//         trigger: "item",
//         formatter: "{a} <br/>{b}: {c} ({d}%)",
//       },
//       legend: {
//         orient: "vertical",
//         right: 10,
//         top: "center",
//         data: ["High", "Medium-High", "Medium-Low", "Low"],
//       },
//       series: [
//         {
//           name: "Performance Distribution",
//           type: "pie",
//           radius: ["50%", "70%"],
//           avoidLabelOverlap: false,
//           itemStyle: {
//             borderRadius: 10,
//             borderColor: "#fff",
//             borderWidth: 2,
//           },
//           label: {
//             show: false,
//             position: "center",
//           },
//           emphasis: {
//             label: {
//               show: true,
//               fontSize: "18",
//               fontWeight: "bold",
//             },
//           },
//           labelLine: {
//             show: false,
//           },
//           data: data,
//           color: ["#28a745", "#20c997", "#fd7e14", "#dc3545"],
//         },
//       ],
//     }
  
//     chart.setOption(option)
  
//     // Resize chart on window resize
//     window.addEventListener("resize", () => {
//       chart.resize()
//     })
//   }
  
//   // Create feature importance chart
//   function createFeatureImportanceChart(data) {
//     const chartElement = document.getElementById("feature-importance-chart")
//     const chart = echarts.init(chartElement)
  
//     const option = {
//       tooltip: {
//         trigger: "axis",
//         axisPointer: {
//           type: "shadow",
//         },
//       },
//       grid: {
//         left: "3%",
//         right: "4%",
//         bottom: "3%",
//         containLabel: true,
//       },
//       xAxis: {
//         type: "value",
//         boundaryGap: [0, 0.01],
//       },
//       yAxis: {
//         type: "category",
//         data: data.features.slice(0, 5),
//       },
//       series: [
//         {
//           name: "Importance",
//           type: "bar",
//           data: data.importance.slice(0, 5),
//           itemStyle: {
//             color: (params) => {
//               const colorList = ["#c23531", "#2f4554", "#61a0a8", "#d48265", "#91c7ae"]
//               return colorList[params.dataIndex % colorList.length]
//             },
//           },
//         },
//       ],
//     }
  
//     chart.setOption(option)
  
//     // Resize chart on window resize
//     window.addEventListener("resize", () => {
//       chart.resize()
//     })
//   }
  
//   // Create detailed feature importance chart
//   function createFeatureImportanceDetailedChart(data) {
//     const chartElement = document.getElementById("feature-importance-detailed-chart")
//     if (!chartElement) return
  
//     // Clear previous chart if it exists
//     echarts.dispose(chartElement)
//     const chart = echarts.init(chartElement)
  
//     const option = {
//       tooltip: {
//         trigger: "axis",
//         axisPointer: {
//           type: "shadow",
//         },
//       },
//       legend: {
//         data: ["Feature Importance"],
//         textStyle: {
//           fontSize: 14,
//           fontWeight: "bold",
//         },
//       },
//       grid: {
//         left: "3%",
//         right: "4%",
//         bottom: "3%",
//         containLabel: true,
//       },
//       xAxis: {
//         type: "value",
//         boundaryGap: [0, 0.01],
//         axisLabel: {
//           fontSize: 12,
//           fontWeight: "bold",
//         },
//       },
//       yAxis: {
//         type: "category",
//         data: data.features,
//         axisLabel: {
//           fontSize: 12,
//           fontWeight: "bold",
//         },
//       },
//       series: [
//         {
//           name: "Feature Importance",
//           type: "bar",
//           data: data.importance,
//           itemStyle: {
//             color: (params) => {
//               // More vibrant color palette
//               const colorList = [
//                 "#FF5733", // Bright red-orange
//                 "#33A8FF", // Bright blue
//                 "#33FF57", // Bright green
//                 "#FF33A8", // Bright pink
//                 "#A833FF", // Bright purple
//                 "#FFD433", // Bright yellow
//                 "#33FFD4", // Bright teal
//                 "#FF8333", // Bright orange
//                 "#3357FF", // Bright indigo
//                 "#FF3333", // Bright red
//               ]
//               return colorList[params.dataIndex % colorList.length]
//             },
//           },
//           label: {
//             show: true,
//             position: "right",
//             formatter: "{c}",
//             fontSize: 12,
//             fontWeight: "bold",
//           },
//         },
//       ],
//     }
  
//     chart.setOption(option)
  
//     // Resize chart on window resize
//     window.addEventListener("resize", () => {
//       chart.resize()
//     })
//   }
  
//   // Create metrics by performance chart
//   function createMetricsByPerformanceChart(metricsData) {
//     const chartElement = document.getElementById("metrics-by-performance-chart")
//     if (!chartElement) return
  
//     // Clear previous chart if it exists
//     echarts.dispose(chartElement)
//     const chart = echarts.init(chartElement)
  
//     // Extract performance classes
//     const performanceClasses = Object.keys(metricsData.agent_age || {})
  
//     // Extract metrics
//     const metrics = ["agent_age", "tenure_months", "unique_proposal", "unique_quotations", "unique_customers"]
  
//     // Prepare series data
//     const series = metrics.map((metric, index) => {
//       const data = performanceClasses.map((pc) => metricsData[metric]?.[pc] || 0)
  
//       // Vibrant colors for each metric
//       const colors = [
//         "#FF5733", // Bright red-orange
//         "#33A8FF", // Bright blue
//         "#33FF57", // Bright green
//         "#FF33A8", // Bright pink
//         "#A833FF", // Bright purple
//       ]
  
//       return {
//         name: formatMetricName(metric),
//         type: "bar",
//         data: data,
//         itemStyle: {
//           color: colors[index % colors.length],
//         },
//         label: {
//           show: true,
//           position: "top",
//           formatter: "{c}",
//           fontSize: 12,
//           fontWeight: "bold",
//         },
//       }
//     })
  
//     const option = {
//       tooltip: {
//         trigger: "axis",
//         axisPointer: {
//           type: "shadow",
//         },
//       },
//       legend: {
//         data: metrics.map(formatMetricName),
//         textStyle: {
//           fontSize: 14,
//           fontWeight: "bold",
//         },
//       },
//       grid: {
//         left: "3%",
//         right: "4%",
//         bottom: "3%",
//         containLabel: true,
//       },
//       xAxis: {
//         type: "category",
//         data: performanceClasses,
//         axisLabel: {
//           fontSize: 12,
//           fontWeight: "bold",
//         },
//       },
//       yAxis: {
//         type: "value",
//         axisLabel: {
//           fontSize: 12,
//           fontWeight: "bold",
//         },
//       },
//       series: series,
//     }
  
//     chart.setOption(option)
  
//     // Resize chart on window resize
//     window.addEventListener("resize", () => {
//       chart.resize()
//     })
  
//     // Helper function to format metric names
//     function formatMetricName(metric) {
//       return metric
//         .split("_")
//         .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
//         .join(" ")
//     }
//   }
  
//   // Create prediction distribution chart
//   function createPredictionDistributionChart(agents) {
//     const chartElement = document.getElementById("prediction-distribution-chart")
//     const chart = echarts.init(chartElement)
  
//     // Extract NILL probabilities
//     const probabilities = agents.map((agent) => agent.nill_probability)
  
//     // Create histogram data
//     const histogramData = []
//     const binSize = 0.1
  
//     for (let i = 0; i < 1; i += binSize) {
//       const binCount = probabilities.filter((p) => p >= i && p < i + binSize).length
//       histogramData.push([i, binCount])
//     }
  
//     const option = {
//       title: {
//         text: "Distribution of NILL Probabilities",
//         left: "center",
//       },
//       tooltip: {
//         trigger: "axis",
//         formatter: (params) => {
//           const x = params[0].data[0]
//           const count = params[0].data[1]
//           return `Probability: ${(x * 100).toFixed(0)}-${((x + binSize) * 100).toFixed(0)}%<br/>Count: ${count}`
//         },
//       },
//       xAxis: {
//         type: "value",
//         name: "NILL Probability",
//         min: 0,
//         max: 1,
//         axisLabel: {
//           formatter: "{value}",
//         },
//       },
//       yAxis: {
//         type: "value",
//         name: "Number of Agents",
//       },
//       series: [
//         {
//           name: "NILL Probability",
//           type: "bar",
//           data: histogramData,
//           itemStyle: {
//             color: (params) => {
//               const value = params.data[0]
//               if (value < 0.25) return "#28a745"
//               if (value < 0.5) return "#ffc107"
//               if (value < 0.75) return "#fd7e14"
//               return "#dc3545"
//             },
//           },
//         },
//       ],
//     }
  
//     chart.setOption(option)
  
//     // Resize chart on window resize
//     window.addEventListener("resize", () => {
//       chart.resize()
//     })
//   }
  
//   // Declare bootstrap and echarts at the top of the file
//   const bootstrap = window.bootstrap
//   const echarts = window.echarts
document.addEventListener("DOMContentLoaded", () => {
    // Initialize navigation
    initNavigation()
  
    // Load dashboard data
    loadDashboardStats()
  
    // Load agents data
    loadAgents()
  
    // Initialize prediction form
    initPredictionForm()
  
    // Initialize agent selector
    initAgentSelector()
  })
  
  // Navigation between sections
  function initNavigation() {
    const navLinks = document.querySelectorAll(".sidebar-menu .nav-link")
  
    navLinks.forEach((link) => {
      link.addEventListener("click", function (e) {
        e.preventDefault()
  
        // Remove active class from all links
        navLinks.forEach((l) => l.classList.remove("active"))
  
        // Add active class to clicked link
        this.classList.add("active")
  
        // Hide all content sections
        document.querySelectorAll(".content-section").forEach((section) => {
          section.classList.remove("active")
        })
  
        // Show the selected content section
        const sectionId = this.getAttribute("data-section")
        document.getElementById(sectionId).classList.add("active")
  
        // Refresh charts if switching to factors tab
        if (sectionId === "factors") {
          // Reload feature importance data
          fetch("/api/feature-importance")
            .then((response) => response.json())
            .then((data) => {
              createFeatureImportanceDetailedChart(data)
            })
            .catch((error) => console.error("Error loading feature importance:", error))
  
          // Reload metrics data
          fetch("/api/dashboard-stats")
            .then((response) => response.json())
            .then((data) => {
              createMetricsByPerformanceChart(data.avg_metrics_by_performance)
            })
            .catch((error) => console.error("Error loading dashboard stats:", error))
        }
  
        // Refresh prediction chart if switching to predictions tab
        if (sectionId === "predictions") {
          fetch("/api/agents")
            .then((response) => response.json())
            .then((data) => {
              createPredictionDistributionChart(data)
            })
            .catch((error) => console.error("Error loading agents for prediction chart:", error))
        }
      })
    })
  }
  
  // Load dashboard statistics
  function loadDashboardStats() {
    fetch("/api/dashboard-stats")
      .then((response) => response.json())
      .then((data) => {
        // Update total agents
        document.getElementById("total-agents").textContent = data.total_agents
  
        // Update risk distribution
        const highRisk = data.risk_distribution["High Risk"] || 0
        const mediumHighRisk = data.risk_distribution["Medium-High Risk"] || 0
        const mediumLowRisk = data.risk_distribution["Medium-Low Risk"] || 0
        const lowRisk = data.risk_distribution["Low Risk"] || 0
  
        document.getElementById("high-risk-agents").textContent = highRisk
        document.getElementById("medium-risk-agents").textContent = mediumHighRisk + mediumLowRisk
        document.getElementById("low-risk-agents").textContent = lowRisk
  
        // Create risk distribution chart
        createRiskDistributionChart(data.risk_distribution)
  
        // Create performance distribution chart
        createPerformanceDistributionChart(data.performance_distribution)
  
        // Create metrics by performance chart - ensure this runs
        setTimeout(() => {
          createMetricsByPerformanceChart(data.avg_metrics_by_performance)
        }, 500)
      })
      .catch((error) => console.error("Error loading dashboard stats:", error))
  
    // Load feature importance
    fetch("/api/feature-importance")
      .then((response) => response.json())
      .then((data) => {
        // Create feature importance charts
        createFeatureImportanceChart(data)
  
        // Ensure this runs with a slight delay to make sure the DOM is ready
        setTimeout(() => {
          createFeatureImportanceDetailedChart(data)
        }, 500)
      })
      .catch((error) => console.error("Error loading feature importance:", error))
  }
  
  // Load agents data
  function loadAgents() {
    fetch("/api/agents")
      .then((response) => response.json())
      .then((data) => {
        // Populate agents table
        populateAgentsTable(data)
  
        // Populate agent selector
        populateAgentSelector(data)
  
        // Create prediction distribution chart with a slight delay to ensure DOM is ready
        setTimeout(() => {
          createPredictionDistributionChart(data)
        }, 500)
      })
      .catch((error) => console.error("Error loading agents:", error))
  }
  
  // Populate agents table
  function populateAgentsTable(agents) {
    const tableBody = document.querySelector("#agents-table tbody")
    tableBody.innerHTML = ""
  
    agents.forEach((agent) => {
      const row = document.createElement("tr")
  
      // Create risk badge class
      let riskBadgeClass = ""
      switch (agent.risk_segment) {
        case "High Risk":
          riskBadgeClass = "badge-high-risk"
          break
        case "Medium-High Risk":
          riskBadgeClass = "badge-medium-high-risk"
          break
        case "Medium-Low Risk":
          riskBadgeClass = "badge-medium-low-risk"
          break
        case "Low Risk":
          riskBadgeClass = "badge-low-risk"
          break
      }
  
      // Create performance badge class
      let performanceBadgeClass = ""
      switch (agent.performance_class) {
        case "High":
          performanceBadgeClass = "badge-high"
          break
        case "Medium-High":
          performanceBadgeClass = "badge-medium-high"
          break
        case "Medium-Low":
          performanceBadgeClass = "badge-medium-low"
          break
        case "Low":
          performanceBadgeClass = "badge-low"
          break
      }
  
      row.innerHTML = `
              <td>${agent.agent_code}</td>
              <td>${agent.agent_name}</td>
              <td>${agent.agent_age}</td>
              <td>${agent.tenure_months}</td>
              <td><span class="badge ${performanceBadgeClass}">${agent.performance_class}</span></td>
              <td><span class="badge ${riskBadgeClass}">${agent.risk_segment}</span></td>
              <td>${(agent.nill_probability * 100).toFixed(2)}%</td>
              <td>
                  <button class="btn btn-sm btn-primary view-agent-btn" data-agent-code="${agent.agent_code}">
                      View
                  </button>
              </td>
          `
  
      tableBody.appendChild(row)
    })
  
    // Add event listeners to view buttons
    document.querySelectorAll(".view-agent-btn").forEach((button) => {
      button.addEventListener("click", function () {
        const agentCode = this.getAttribute("data-agent-code")
        openAgentDetailModal(agentCode)
      })
    })
  }
  
  // Populate agent selector
  function populateAgentSelector(agents) {
    const selector = document.getElementById("agent-select")
  
    // Clear existing options except the first one
    while (selector.options.length > 1) {
      selector.remove(1)
    }
  
    // Add agents to selector
    agents.forEach((agent) => {
      const option = document.createElement("option")
      option.value = agent.agent_code
      option.textContent = `${agent.agent_code} - ${agent.agent_name} (${agent.performance_class})`
      selector.appendChild(option)
    })
  }
  
  // Initialize prediction form
  function initPredictionForm() {
    const form = document.getElementById("prediction-form")
  
    form.addEventListener("submit", (e) => {
      e.preventDefault()
  
      // Show loading indicator
      const submitBtn = form.querySelector('button[type="submit"]')
      const originalBtnText = submitBtn.innerHTML
      submitBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...'
      submitBtn.disabled = true
  
      // Get form data
      const formData = {
        agent_age: Number.parseInt(document.getElementById("agent_age").value),
        tenure_months: Number.parseInt(document.getElementById("tenure_months").value),
        unique_proposal: Number.parseInt(document.getElementById("unique_proposal").value),
        unique_quotations: Number.parseInt(document.getElementById("unique_quotations").value),
        unique_customers: Number.parseInt(document.getElementById("unique_customers").value),
        proposal_to_quotation_ratio: Number.parseFloat(document.getElementById("proposal_to_quotation_ratio").value),
      }
  
      // Calculate derived features
      formData.proposal_intensity = formData.unique_proposal / Math.max(1, formData.tenure_months)
      formData.quotation_intensity = formData.unique_quotations / Math.max(1, formData.tenure_months)
      formData.customer_intensity = formData.unique_customers / Math.max(1, formData.tenure_months)
  
      // Make prediction
      fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`)
          }
          return response.json()
        })
        .then((data) => {
          // Reset button
          submitBtn.innerHTML = originalBtnText
          submitBtn.disabled = false
  
          if (data.error) {
            alert("Error: " + data.error)
            return
          }
  
          // Show prediction result
          document.getElementById("prediction-result").style.display = "block"
          document.getElementById("prediction-probability").textContent = (data.nill_probability * 100).toFixed(2) + "%"
          document.getElementById("prediction-risk").textContent = data.risk_segment
          document.getElementById("prediction-performance").textContent = data.performance_class
  
          // Set alert class based on risk
          const alertElement = document.getElementById("prediction-alert")
          alertElement.className = "alert"
  
          switch (data.risk_segment) {
            case "High Risk":
              alertElement.classList.add("alert-danger")
              break
            case "Medium-High Risk":
              alertElement.classList.add("alert-warning")
              break
            case "Medium-Low Risk":
              alertElement.classList.add("alert-info")
              break
            case "Low Risk":
              alertElement.classList.add("alert-success")
              break
          }
  
          // Show recommendations
          const actionsList = document.getElementById("prediction-actions")
          actionsList.innerHTML = ""
  
          data.recommendations.forEach((recommendation) => {
            const li = document.createElement("li")
            li.textContent = recommendation
            actionsList.appendChild(li)
          })
  
          // Scroll to the prediction result
          document.getElementById("prediction-result").scrollIntoView({ behavior: "smooth", block: "start" })
        })
        .catch((error) => {
          // Reset button
          submitBtn.innerHTML = originalBtnText
          submitBtn.disabled = false
  
          console.error("Error making prediction:", error)
          alert("Error making prediction. Please try again.")
        })
    })
  }
  
  // Initialize agent selector
  function initAgentSelector() {
    const selector = document.getElementById("agent-select")
  
    selector.addEventListener("change", function () {
      const agentCode = this.value
  
      if (!agentCode) {
        // Hide agent details if no agent is selected
        document.getElementById("agent-details").style.display = "none"
        document.getElementById("agent-factors").style.display = "none"
        return
      }
  
      // Load agent details
      fetch(`/api/agent/${agentCode}`)
        .then((response) => response.json())
        .then((data) => {
          // Show agent details
          document.getElementById("agent-details").style.display = "flex"
          document.getElementById("agent-factors").style.display = "block"
  
          // Update agent profile
          document.getElementById("detail-agent-code").textContent = data.agent_code
          document.getElementById("detail-agent-name").textContent = data.agent_name
          document.getElementById("detail-agent-age").textContent = data.agent_age
          document.getElementById("detail-agent-tenure").textContent = data.tenure_months
          document.getElementById("detail-performance-class").textContent = data.performance_class
          document.getElementById("detail-risk-level").textContent = data.risk_segment
          document.getElementById("detail-nill-probability").textContent = (data.nill_probability * 100).toFixed(2) + "%"
  
          // Update recommendations
          const recommendationList = document.getElementById("recommendation-list")
          recommendationList.innerHTML = ""
  
          data.recommendations.forEach((recommendation) => {
            const li = document.createElement("li")
            li.textContent = recommendation
            recommendationList.appendChild(li)
          })
  
          // Update factors
          const factorsList = document.getElementById("agent-factors-list")
          factorsList.innerHTML = ""
  
          data.top_factors.forEach((factor) => {
            const factorCard = document.createElement("div")
            factorCard.className = `factor-card ${factor.impact}-impact`
  
            const factorTitle = document.createElement("div")
            factorTitle.className = "factor-title"
            factorTitle.textContent = factor.factor
  
            const factorDescription = document.createElement("div")
            factorDescription.className = "factor-description"
            factorDescription.textContent = factor.description
  
            const factorImpact = document.createElement("span")
            factorImpact.className = `badge badge-impact-${factor.impact}`
            factorImpact.textContent = factor.impact.charAt(0).toUpperCase() + factor.impact.slice(1) + " Impact"
  
            factorCard.appendChild(factorTitle)
            factorCard.appendChild(factorDescription)
            factorCard.appendChild(document.createElement("br"))
            factorCard.appendChild(factorImpact)
  
            factorsList.appendChild(factorCard)
          })
        })
        .catch((error) => console.error("Error loading agent details:", error))
    })
  }
  
  // Open agent detail modal
  function openAgentDetailModal(agentCode) {
    fetch(`/api/agent/${agentCode}`)
      .then((response) => response.json())
      .then((data) => {
        // Update modal title
        document.getElementById("agentDetailModalLabel").textContent =
          `Agent Details: ${data.agent_name} (${data.agent_code})`
  
        // Update agent profile
        document.getElementById("modal-agent-code").textContent = data.agent_code
        document.getElementById("modal-agent-name").textContent = data.agent_name
        document.getElementById("modal-agent-age").textContent = data.agent_age
        document.getElementById("modal-agent-tenure").textContent = data.tenure_months
        document.getElementById("modal-performance-class").textContent = data.performance_class
        document.getElementById("modal-risk-level").textContent = data.risk_segment
        document.getElementById("modal-nill-probability").textContent = (data.nill_probability * 100).toFixed(2) + "%"
  
        // Update agent metrics
        document.getElementById("modal-unique-proposals").textContent = data.unique_proposal
        document.getElementById("modal-unique-quotations").textContent = data.unique_quotations
        document.getElementById("modal-unique-customers").textContent = data.unique_customers
        document.getElementById("modal-proposal-intensity").textContent = data.proposal_intensity
          ? data.proposal_intensity.toFixed(2)
          : "N/A"
        document.getElementById("modal-quotation-intensity").textContent = data.quotation_intensity
          ? data.quotation_intensity.toFixed(2)
          : "N/A"
        document.getElementById("modal-customer-intensity").textContent = data.customer_intensity
          ? data.customer_intensity.toFixed(2)
          : "N/A"
        document.getElementById("modal-proposal-quotation-ratio").textContent = data.proposal_to_quotation_ratio
          ? data.proposal_to_quotation_ratio.toFixed(2)
          : "N/A"
  
        // Update recommendations
        const recommendationList = document.getElementById("modal-recommendation-list")
        recommendationList.innerHTML = ""
  
        data.recommendations.forEach((recommendation) => {
          const li = document.createElement("li")
          li.textContent = recommendation
          recommendationList.appendChild(li)
        })
  
        // Update factors
        const factorsList = document.getElementById("modal-agent-factors")
        factorsList.innerHTML = ""
  
        data.top_factors.forEach((factor) => {
          const factorCard = document.createElement("div")
          factorCard.className = `factor-card ${factor.impact}-impact`
  
          const factorTitle = document.createElement("div")
          factorTitle.className = "factor-title"
          factorTitle.textContent = factor.factor
  
          const factorDescription = document.createElement("div")
          factorDescription.className = "factor-description"
          factorDescription.textContent = factor.description
  
          const factorImpact = document.createElement("span")
          factorImpact.className = `badge badge-impact-${factor.impact}`
          factorImpact.textContent = factor.impact.charAt(0).toUpperCase() + factor.impact.slice(1) + " Impact"
  
          factorCard.appendChild(factorTitle)
          factorCard.appendChild(factorDescription)
          factorCard.appendChild(document.createElement("br"))
          factorCard.appendChild(factorImpact)
  
          factorsList.appendChild(factorCard)
        })
  
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById("agentDetailModal"))
        modal.show()
      })
      .catch((error) => console.error("Error loading agent details:", error))
  }
  
  // Create risk distribution chart
  function createRiskDistributionChart(riskDistribution) {
    const chartElement = document.getElementById("risk-distribution-chart")
    const chart = echarts.init(chartElement)
  
    const data = [
      { value: riskDistribution["High Risk"] || 0, name: "High Risk" },
      { value: riskDistribution["Medium-High Risk"] || 0, name: "Medium-High Risk" },
      { value: riskDistribution["Medium-Low Risk"] || 0, name: "Medium-Low Risk" },
      { value: riskDistribution["Low Risk"] || 0, name: "Low Risk" },
    ]
  
    const option = {
      tooltip: {
        trigger: "item",
        formatter: "{a} <br/>{b}: {c} ({d}%)",
      },
      legend: {
        orient: "vertical",
        right: 10,
        top: "center",
        data: ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"],
      },
      series: [
        {
          name: "Risk Distribution",
          type: "pie",
          radius: ["50%", "70%"],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: "#fff",
            borderWidth: 2,
          },
          label: {
            show: false,
            position: "center",
          },
          emphasis: {
            label: {
              show: true,
              fontSize: "18",
              fontWeight: "bold",
            },
          },
          labelLine: {
            show: false,
          },
          data: data,
          color: ["#dc3545", "#fd7e14", "#ffc107", "#28a745"],
        },
      ],
    }
  
    chart.setOption(option)
  
    // Resize chart on window resize
    window.addEventListener("resize", () => {
      chart.resize()
    })
  }
  
  // Create performance distribution chart
  function createPerformanceDistributionChart(performanceDistribution) {
    const chartElement = document.getElementById("performance-distribution-chart")
    const chart = echarts.init(chartElement)
  
    const data = [
      { value: performanceDistribution["High"] || 0, name: "High" },
      { value: performanceDistribution["Medium-High"] || 0, name: "Medium-High" },
      { value: performanceDistribution["Medium-Low"] || 0, name: "Medium-Low" },
      { value: performanceDistribution["Low"] || 0, name: "Low" },
    ]
  
    const option = {
      tooltip: {
        trigger: "item",
        formatter: "{a} <br/>{b}: {c} ({d}%)",
      },
      legend: {
        orient: "vertical",
        right: 10,
        top: "center",
        data: ["High", "Medium-High", "Medium-Low", "Low"],
      },
      series: [
        {
          name: "Performance Distribution",
          type: "pie",
          radius: ["50%", "70%"],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: "#fff",
            borderWidth: 2,
          },
          label: {
            show: false,
            position: "center",
          },
          emphasis: {
            label: {
              show: true,
              fontSize: "18",
              fontWeight: "bold",
            },
          },
          labelLine: {
            show: false,
          },
          data: data,
          color: ["#28a745", "#20c997", "#fd7e14", "#dc3545"],
        },
      ],
    }
  
    chart.setOption(option)
  
    // Resize chart on window resize
    window.addEventListener("resize", () => {
      chart.resize()
    })
  }
  
  // Create feature importance chart
  function createFeatureImportanceChart(data) {
    const chartElement = document.getElementById("feature-importance-chart")
    const chart = echarts.init(chartElement)
  
    const option = {
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        containLabel: true,
      },
      xAxis: {
        type: "value",
        boundaryGap: [0, 0.01],
      },
      yAxis: {
        type: "category",
        data: data.features.slice(0, 5),
      },
      series: [
        {
          name: "Importance",
          type: "bar",
          data: data.importance.slice(0, 5),
          itemStyle: {
            color: (params) => {
              const colorList = ["#c23531", "#2f4554", "#61a0a8", "#d48265", "#91c7ae"]
              return colorList[params.dataIndex % colorList.length]
            },
          },
        },
      ],
    }
  
    chart.setOption(option)
  
    // Resize chart on window resize
    window.addEventListener("resize", () => {
      chart.resize()
    })
  }
  
  // Create detailed feature importance chart
  function createFeatureImportanceDetailedChart(data) {
    const chartElement = document.getElementById("feature-importance-detailed-chart")
    if (!chartElement) return
  
    // Clear previous chart if it exists
    echarts.dispose(chartElement)
    const chart = echarts.init(chartElement)
  
    const option = {
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
      },
      legend: {
        data: ["Feature Importance"],
        textStyle: {
          fontSize: 14,
          fontWeight: "bold",
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        containLabel: true,
      },
      xAxis: {
        type: "value",
        boundaryGap: [0, 0.01],
        axisLabel: {
          fontSize: 12,
          fontWeight: "bold",
        },
      },
      yAxis: {
        type: "category",
        data: data.features,
        axisLabel: {
          fontSize: 12,
          fontWeight: "bold",
        },
      },
      series: [
        {
          name: "Feature Importance",
          type: "bar",
          data: data.importance,
          itemStyle: {
            color: (params) => {
              // More vibrant color palette
              const colorList = [
                "#FF5733", // Bright red-orange
                "#33A8FF", // Bright blue
                "#33FF57", // Bright green
                "#FF33A8", // Bright pink
                "#A833FF", // Bright purple
                "#FFD433", // Bright yellow
                "#33FFD4", // Bright teal
                "#FF8333", // Bright orange
                "#3357FF", // Bright indigo
                "#FF3333", // Bright red
              ]
              return colorList[params.dataIndex % colorList.length]
            },
          },
          label: {
            show: true,
            position: "right",
            formatter: "{c}",
            fontSize: 12,
            fontWeight: "bold",
          },
        },
      ],
    }
  
    chart.setOption(option)
  
    // Resize chart on window resize
    window.addEventListener("resize", () => {
      chart.resize()
    })
  }
  
  // Create metrics by performance chart
  function createMetricsByPerformanceChart(metricsData) {
    const chartElement = document.getElementById("metrics-by-performance-chart")
    if (!chartElement) return
  
    // Clear previous chart if it exists
    echarts.dispose(chartElement)
    const chart = echarts.init(chartElement)
  
    // Extract performance classes
    const performanceClasses = Object.keys(metricsData.agent_age || {})
  
    // Extract metrics
    const metrics = ["agent_age", "tenure_months", "unique_proposal", "unique_quotations", "unique_customers"]
  
    // Prepare series data
    const series = metrics.map((metric, index) => {
      const data = performanceClasses.map((pc) => metricsData[metric]?.[pc] || 0)
  
      // Vibrant colors for each metric
      const colors = [
        "#FF5733", // Bright red-orange
        "#33A8FF", // Bright blue
        "#33FF57", // Bright green
        "#FF33A8", // Bright pink
        "#A833FF", // Bright purple
      ]
  
      return {
        name: formatMetricName(metric),
        type: "bar",
        data: data,
        itemStyle: {
          color: colors[index % colors.length],
        },
        label: {
          show: true,
          position: "top",
          formatter: "{c}",
          fontSize: 12,
          fontWeight: "bold",
        },
      }
    })
  
    const option = {
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
      },
      legend: {
        data: metrics.map(formatMetricName),
        textStyle: {
          fontSize: 14,
          fontWeight: "bold",
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: performanceClasses,
        axisLabel: {
          fontSize: 12,
          fontWeight: "bold",
        },
      },
      yAxis: {
        type: "value",
        axisLabel: {
          fontSize: 12,
          fontWeight: "bold",
        },
      },
      series: series,
    }
  
    chart.setOption(option)
  
    // Resize chart on window resize
    window.addEventListener("resize", () => {
      chart.resize()
    })
  
    // Helper function to format metric names
    function formatMetricName(metric) {
      return metric
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
    }
  }
  
  // Create prediction distribution chart
  function createPredictionDistributionChart(agents) {
    const chartElement = document.getElementById("prediction-distribution-chart")
    if (!chartElement) return
  
    // Clear previous chart if it exists
    echarts.dispose(chartElement)
    const chart = echarts.init(chartElement)
  
    // Extract NILL probabilities
    const probabilities = agents.map((agent) => agent.nill_probability)
  
    // Create histogram data
    const histogramData = []
    const binSize = 0.1
  
    for (let i = 0; i < 1; i += binSize) {
      const binCount = probabilities.filter((p) => p >= i && p < i + binSize).length
      histogramData.push([i, binCount])
    }
  
    const option = {
      title: {
        text: "Distribution of NILL Probabilities",
        left: "center",
        textStyle: {
          fontSize: 18,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        formatter: (params) => {
          const x = params[0].data[0]
          const count = params[0].data[1]
          return `Probability: ${(x * 100).toFixed(0)}-${((x + binSize) * 100).toFixed(0)}%<br/>Count: ${count}`
        },
      },
      xAxis: {
        type: "value",
        name: "NILL Probability",
        nameTextStyle: {
          fontSize: 14,
          fontWeight: "bold",
        },
        min: 0,
        max: 1,
        axisLabel: {
          formatter: "{value}",
          fontSize: 12,
          fontWeight: "bold",
        },
      },
      yAxis: {
        type: "value",
        name: "Number of Agents",
        nameTextStyle: {
          fontSize: 14,
          fontWeight: "bold",
        },
        axisLabel: {
          fontSize: 12,
          fontWeight: "bold",
        },
      },
      series: [
        {
          name: "NILL Probability",
          type: "bar",
          data: histogramData,
          itemStyle: {
            color: (params) => {
              const value = params.data[0]
              if (value < 0.25) return "#2ecc71" // Green
              if (value < 0.5) return "#f39c12" // Yellow
              if (value < 0.75) return "#e67e22" // Orange
              return "#e74c3c" // Red
            },
          },
          label: {
            show: true,
            position: "top",
            formatter: "{c}",
            fontSize: 12,
            fontWeight: "bold",
          },
        },
      ],
    }
  
    chart.setOption(option)
  
    // Resize chart on window resize
    window.addEventListener("resize", () => {
      chart.resize()
    })
  }
  
  // Declare bootstrap and echarts at the top of the file
  const bootstrap = window.bootstrap
  const echarts = window.echarts
  