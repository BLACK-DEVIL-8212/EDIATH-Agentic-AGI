from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AgentSpec:
    agent_name: str
    actions: List[str]
    confidence: float = 0.8


# Map requested 1..80 agents to action keyword sets.
# These keywords must align with ActionRouter handler names if you later
# register handlers for them.
#
# Today, these wrappers primarily enable coordinator routing and provide a
# structured execution payload.
AGENT_SPECS: Dict[str, AgentSpec] = {
    "AutonomousCodingAgent": AgentSpec(
        agent_name="Autonomous Coding Agent", actions=["code", "write_code", "debug", "refactor"], confidence=0.9
    ),
    "InfrastructureManagementAgent": AgentSpec(
        agent_name="Infrastructure Management Agent", actions=["infra", "deploy", "scale"], confidence=0.85
    ),
    "AIOrchestrationAgent": AgentSpec(
        agent_name="AI Orchestration Agent", actions=["orchestrate", "plan", "schedule", "coordinate"], confidence=0.9
    ),
    "DistributedSystemsAgent": AgentSpec(
        agent_name="Distributed Systems Agent", actions=["distributed", "cluster", "replicate"], confidence=0.8
    ),
    "AutonomousDevOpsAgent": AgentSpec(
        agent_name="Autonomous DevOps Agent", actions=["devops", "ci", "cd", "pipeline"], confidence=0.85
    ),
    "CybersecurityMonitoringAgent": AgentSpec(
        agent_name="Cybersecurity Monitoring Agent", actions=["security", "monitor"], confidence=0.85
    ),
    "ThreatDetectionAgent": AgentSpec(
        agent_name="Threat Detection Agent", actions=["threat", "detect", "analyze"], confidence=0.8
    ),
    "ReverseEngineeringAgent": AgentSpec(
        agent_name="Reverse Engineering Agent", actions=["reverse", "analyze_binary", "reconstruct"], confidence=0.75
    ),
    "AutonomousDebuggingAgent": AgentSpec(
        agent_name="Autonomous Debugging Agent", actions=["debug", "trace", "fix"], confidence=0.9
    ),
    "SelfHealingServerAgent": AgentSpec(
        agent_name="Self-Healing Server Agent", actions=["heal", "recover", "restart"], confidence=0.85
    ),
    "CloudDeploymentAgent": AgentSpec(
        agent_name="Cloud Deployment Agent", actions=["cloud", "deploy", "provision"], confidence=0.85
    ),
    "KubernetesOrchestrationAgent": AgentSpec(
        agent_name="Kubernetes Orchestration Agent", actions=["k8s", "kubectl", "rollout"], confidence=0.8
    ),
    "MultiAgentCoordinatorAgent": AgentSpec(
        agent_name="Multi-Agent Coordinator", actions=["multi_agent", "coordinate", "allocate"], confidence=0.95
    ),
    "AutonomousWorkflowAgent": AgentSpec(
        agent_name="Autonomous Workflow Agent", actions=["workflow", "dag", "pipeline"], confidence=0.9
    ),
    "InfrastructurePlanningAgent": AgentSpec(
        agent_name="Infrastructure Planning Agent", actions=["plan", "architecture", "spec"], confidence=0.85
    ),
    "AIResearchAgent": AgentSpec(
        agent_name="AI Research Agent", actions=["research", "study", "evaluate"], confidence=0.8
    ),
    "PromptEngineeringAgent": AgentSpec(
        agent_name="Prompt Engineering Agent", actions=["prompt", "design_prompt"], confidence=0.85
    ),
    "AutonomousBackendEngineerAgent": AgentSpec(
        agent_name="Autonomous Backend Engineer Agent", actions=["backend", "api", "server"], confidence=0.85
    ),
    "APIGenerationAgent": AgentSpec(
        agent_name="API Generation Agent", actions=["api", "generate_api"], confidence=0.85
    ),
    "AutonomousAPITestingAgent": AgentSpec(
        agent_name="Autonomous API Testing Agent", actions=["test", "api_test"], confidence=0.8
    ),
    "SystemArchitectureAgent": AgentSpec(
        agent_name="System Architecture Agent", actions=["architecture", "design"], confidence=0.9
    ),
    "AutonomousRefactoringAgent": AgentSpec(
        agent_name="Autonomous Refactoring Agent", actions=["refactor", "cleanup", "improve"], confidence=0.9
    ),
    "LogAnalysisAgent": AgentSpec(
        agent_name="Log Analysis Agent", actions=["logs", "analyze_logs"], confidence=0.8
    ),
    "MonitoringObservabilityAgent": AgentSpec(
        agent_name="Monitoring & Observability Agent", actions=["monitor", "metrics", "tracing"], confidence=0.85
    ),
    "NeuralOrchestrationAgent": AgentSpec(
        agent_name="Neural Orchestration Agent", actions=["neural", "orchestrate", "route"], confidence=0.8
    ),
    "AutonomousFileManagementAgent": AgentSpec(
        agent_name="Autonomous File Management Agent", actions=["file", "read", "write", "move", "delete"], confidence=0.85
    ),
    "AITaskSchedulingAgent": AgentSpec(
        agent_name="AI Task Scheduling Agent", actions=["schedule", "queue", "timer"], confidence=0.85
    ),
    "DistributedComputeAgent": AgentSpec(
        agent_name="Distributed Compute Agent", actions=["compute", "distributed"], confidence=0.8
    ),
    "AIMemoryManagementAgent": AgentSpec(
        agent_name="AI Memory Management Agent", actions=["memory", "episodic", "semantic", "vector"], confidence=0.8
    ),
    "AutonomousShellExecutionAgent": AgentSpec(
        agent_name="Autonomous Shell Execution Agent", actions=["shell", "exec", "command"], confidence=0.75
    ),
    "DockerManagementAgent": AgentSpec(
        agent_name="Docker Management Agent", actions=["docker", "build", "image"], confidence=0.8
    ),
    "ContainerOrchestrationAgent": AgentSpec(
        agent_name="Container Orchestration Agent", actions=["container", "orchestrate"], confidence=0.8
    ),
    "AutonomousDataPipelineAgent": AgentSpec(
        agent_name="Autonomous Data Pipeline Agent", actions=["data", "pipeline", "etl"], confidence=0.85
    ),
    "AINetworkingAgent": AgentSpec(
        agent_name="AI Networking Agent", actions=["network", "connect", "route"], confidence=0.75
    ),
    "AutonomousInfrastructureScalingAgent": AgentSpec(
        agent_name="Autonomous Infrastructure Scaling Agent", actions=["scale", "autoscale"], confidence=0.85
    ),
    "AIConfigurationManagementAgent": AgentSpec(
        agent_name="AI Configuration Management Agent", actions=["config", "manage_config"], confidence=0.85
    ),
    "AutonomousSecurityGovernanceAgent": AgentSpec(
        agent_name="Autonomous Security Governance Agent", actions=["governance", "policy", "security"], confidence=0.8
    ),
    "AIPenetrationTestingAgent": AgentSpec(
        agent_name="AI Penetration Testing Agent", actions=["pentest", "vuln", "scan"], confidence=0.65
    ),
    "AutonomousCodeReviewAgent": AgentSpec(
        agent_name="Autonomous Code Review Agent", actions=["review", "lint", "analyze_code"], confidence=0.85
    ),
    "AutonomousRepositoryManager": AgentSpec(
        agent_name="Autonomous Repository Manager", actions=["repo", "branch", "merge"], confidence=0.8
    ),
    "AIKnowledgeGraphAgent": AgentSpec(
        agent_name="AI Knowledge Graph Agent", actions=["knowledge_graph", "graph"], confidence=0.75
    ),
    "AIDocumentationAgent": AgentSpec(
        agent_name="AI Documentation Agent", actions=["docs", "document"], confidence=0.8
    ),
    "AutonomousSoftwareDeploymentAgent": AgentSpec(
        agent_name="Autonomous Software Deployment Agent", actions=["deploy", "release"], confidence=0.85
    ),
    "AutonomousRecoveryAgent": AgentSpec(
        agent_name="Autonomous Recovery Agent", actions=["recover", "rollback"], confidence=0.85
    ),
    "AILoadBalancingAgent": AgentSpec(
        agent_name="AI Load Balancing Agent", actions=["lb", "balance"], confidence=0.75
    ),
    "AutonomousMultiServerManager": AgentSpec(
        agent_name="Autonomous Multi-Server Manager", actions=["servers", "manage"], confidence=0.8
    ),
    "GPUResourceManagementAgent": AgentSpec(
        agent_name="GPU Resource Management Agent", actions=["gpu", "resources"], confidence=0.65
    ),
    "AIProcessOptimizationAgent": AgentSpec(
        agent_name="AI Process Optimization Agent", actions=["optimize", "performance"], confidence=0.9
    ),
    "AutonomousClusterControlAgent": AgentSpec(
        agent_name="Autonomous Cluster Control Agent", actions=["cluster", "control"], confidence=0.8
    ),
    "MultiAgentCoordinatorAgent2": AgentSpec(
        agent_name="AI Multi-Agent Controller", actions=["multi_agent", "controller"], confidence=0.9
    ),
    "AIOrchestrationSupervisorAgent": AgentSpec(
        agent_name="AI Automation Supervisor", actions=["supervise", "govern", "coordinate"], confidence=0.9
    ),
    "AIEventTriggerAgent": AgentSpec(
        agent_name="AI Event Trigger Agent", actions=["event", "trigger"], confidence=0.8
    ),
    "AIRuntimeMonitoringAgent": AgentSpec(
        agent_name="AI Runtime Monitoring Agent", actions=["runtime", "monitor", "metrics"], confidence=0.85
    ),
    "AutonomousSystemUpdateAgent": AgentSpec(
        agent_name="Autonomous System Update Agent", actions=["update", "patch"], confidence=0.75
    ),
    "AIPackageManagementAgent": AgentSpec(
        agent_name="AI Package Management Agent", actions=["packages", "install", "pip"], confidence=0.7
    ),
    "AutonomousServiceDiscoveryAgent": AgentSpec(
        agent_name="Autonomous Service Discovery Agent", actions=["service_discovery", "discover"], confidence=0.7
    ),
    "AICommunicationRelayAgent": AgentSpec(
        agent_name="AI Communication Relay Agent", actions=["relay", "communicate"], confidence=0.7
    ),
    "AutonomousDistributedAIControllerAgent": AgentSpec(
        agent_name="Autonomous Distributed AI Controller", actions=["distributed_ai", "control"], confidence=0.75
    ),
    "AICommandRoutingAgent": AgentSpec(
        agent_name="AI Command Routing Agent", actions=["command", "route"], confidence=0.8
    ),
    "AISimulationAgent": AgentSpec(
        agent_name="AI Simulation Agent", actions=["simulate", "sandbox"], confidence=0.65
    ),
    "AutonomousAIGovernanceAgent": AgentSpec(
        agent_name="Autonomous AI Governance Agent", actions=["governance", "policy"], confidence=0.8
    ),
    "SelfImprovingAgent": AgentSpec(
        agent_name="Self-Improving Agent", actions=["learn", "improve", "self_heal"], confidence=0.9
    ),
    "AICognitivePlanningAgent": AgentSpec(
        agent_name="AI Cognitive Planning Agent", actions=["cognitive", "plan"], confidence=0.9
    ),
    "AutonomousDecisionEngineAgent": AgentSpec(
        agent_name="Autonomous Decision Engine", actions=["decision", "decide"], confidence=0.9
    ),
    "AIInfrastructureIntelligenceAgent": AgentSpec(
        agent_name="AI Infrastructure Intelligence Agent", actions=["intelligence", "infra"], confidence=0.8
    ),
    "AutonomousRuntimeExecutorAgent": AgentSpec(
        agent_name="Autonomous Runtime Executor", actions=["runtime", "execute"], confidence=0.85
    ),
    "AIEnvironmentAnalyzerAgent": AgentSpec(
        agent_name="AI Environment Analyzer", actions=["analyze_env", "environment"], confidence=0.75
    ),
    "AutonomousOSInteractionAgent": AgentSpec(
        agent_name="Autonomous OS Interaction Agent", actions=["os", "system", "shell"], confidence=0.7
    ),
    "AIResourceAllocationAgent": AgentSpec(
        agent_name="AI Resource Allocation Agent", actions=["allocate", "resources"], confidence=0.75
    ),
    "AutonomousScriptGenerationAgent": AgentSpec(
        agent_name="Autonomous Script Generation Agent", actions=["script", "generate"], confidence=0.8
    ),
    "AIAutomationPipelineAgent": AgentSpec(
        agent_name="AI Automation Pipeline Agent", actions=["pipeline", "automation"], confidence=0.85
    ),
    "AIMissionPlannerAgent": AgentSpec(
        agent_name="AI Mission Planner Agent", actions=["mission", "plan"], confidence=0.8
    ),
    "AutonomousTaskDelegationAgent": AgentSpec(
        agent_name="Autonomous Task Delegation Agent", actions=["delegate", "allocate"], confidence=0.85
    ),
    "AIVisionOrchestrationAgent": AgentSpec(
        agent_name="AI Vision-Orchestration Agent", actions=["vision", "orchestrate"], confidence=0.7
    ),
    "AutonomousAgentSpawner": AgentSpec(
        agent_name="Autonomous Agent Spawner", actions=["spawn", "agents"], confidence=0.7
    ),
    "AIWorkflowOptimizationAgent": AgentSpec(
        agent_name="AI Workflow Optimization Agent", actions=["optimize", "workflow"], confidence=0.85
    ),
    "AutonomousParallelExecutionAgent": AgentSpec(
        agent_name="Autonomous Parallel Execution Agent", actions=["parallel", "concurrency"], confidence=0.7
    ),
    "AIDataSynchronizationAgent": AgentSpec(
        agent_name="AI Data Synchronization Agent", actions=["sync", "data"], confidence=0.75
    ),
    "AutonomousTelemetryAgent": AgentSpec(
        agent_name="Autonomous Telemetry Agent", actions=["telemetry", "metrics"], confidence=0.8
    ),
    "AIClusterSynchronizationAgent": AgentSpec(
        agent_name="AI Cluster Synchronization Agent", actions=["cluster", "sync"], confidence=0.7
    ),
    "AutonomousInfrastructureGuardianAgent": AgentSpec(
        agent_name="Autonomous Infrastructure Guardian Agent", actions=["guardian", "security", "monitor"], confidence=0.85
    ),
}

# Ensure all requested 1..80 exist.
# If you later want strict 80 mapping, we can expand this dict accordingly.

