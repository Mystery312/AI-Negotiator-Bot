# Multi-Department Resource Management System
## Comprehensive Implementation Plan

## 1. System Overview

### 1.1 Vision
A distributed resource management system where autonomous AI chatbots, each representing a different department or sector, negotiate with each other to determine optimal resource allocation. Each chatbot uses a shared base thinking logic but maintains department-specific priorities, constraints, and objectives.

### 1.2 Core Concept
- **Department Chatbots**: Each chatbot represents a department/sector (e.g., Engineering, Marketing, Sales, Operations, HR, Finance)
- **Base Thinking Logic**: Shared reasoning framework that all chatbots use for decision-making
- **Negotiation Protocol**: Structured communication and argumentation system for resource allocation
- **Resource Pool**: Centralized or distributed pool of resources (budget, personnel, equipment, time, etc.)
- **Consensus Mechanism**: Process for reaching agreement on resource splits

### 1.3 Key Objectives
- Fair and efficient resource allocation across departments
- Transparent negotiation process with reasoning trails
- Adaptable to changing priorities and constraints
- Scalable to multiple departments and resource types
- Audit trail for decision-making processes

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Resource Management Hub                   │
│  - Resource Pool Manager                                     │
│  - Negotiation Orchestrator                                  │
│  - Consensus Validator                                       │
│  - Audit Logger                                              │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐
│ Engineering  │   │   Marketing   │   │    Sales     │
│   Chatbot    │   │   Chatbot     │   │   Chatbot    │
└──────────────┘   └───────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼──────┐
                    │  Base Logic  │
                    │   Engine     │
                    └──────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Base Thinking Logic Engine
- **Purpose**: Shared reasoning framework for all department chatbots
- **Responsibilities**:
  - Decision-making algorithms
  - Argument evaluation and generation
  - Resource need assessment
  - Trade-off analysis
  - Strategic reasoning patterns
- **Implementation**: Core module that all department chatbots inherit/extend

#### 2.2.2 Department Chatbots
- **Purpose**: Autonomous agents representing specific departments
- **Responsibilities**:
  - Assess department needs
  - Generate resource requests with justifications
  - Evaluate proposals from other departments
  - Negotiate and counter-propose
  - Accept or reject final allocations
- **Department-Specific Attributes**:
  - Priority weights for different resource types
  - Historical usage patterns
  - Current project requirements
  - Strategic objectives
  - Constraints and dependencies

#### 2.2.3 Resource Management Hub
- **Purpose**: Central coordination and management
- **Components**:
  - **Resource Pool Manager**: Tracks available resources, allocations, and constraints
  - **Negotiation Orchestrator**: Manages negotiation rounds, turn-taking, and protocols
  - **Consensus Validator**: Verifies agreement validity and resource feasibility
  - **Audit Logger**: Records all negotiations, decisions, and reasoning

#### 2.2.4 Communication Layer
- **Purpose**: Inter-chatbot communication infrastructure
- **Features**:
  - Message routing between chatbots
  - Protocol enforcement
  - Message queuing and ordering
  - Conflict resolution mechanisms

---

## 3. Base Thinking Logic Specification

### 3.1 Core Reasoning Framework

#### 3.1.1 Need Assessment
```
Input: Department state, current projects, historical data
Process:
  1. Analyze current resource utilization
  2. Identify upcoming requirements
  3. Calculate resource gaps
  4. Prioritize needs based on:
     - Strategic importance
     - Time sensitivity
     - Dependencies
     - ROI projections
Output: Resource request with justification
```

#### 3.1.2 Argument Generation
```
Input: Resource request, department priorities, constraints
Process:
  1. Identify key justifications:
     - Business impact
     - Strategic alignment
     - Urgency
     - Historical precedent
     - Dependencies on other departments
  2. Structure argument:
     - Claim
     - Evidence
     - Reasoning
     - Counter-arguments to anticipate
Output: Structured argument for resource request
```

#### 3.1.3 Proposal Evaluation
```
Input: Proposal from another department, own needs, resource pool
Process:
  1. Assess impact on own department
  2. Evaluate proposal fairness
  3. Identify trade-offs
  4. Determine acceptance threshold
  5. Generate counter-proposal if needed
Output: Accept, Reject, or Counter-propose with reasoning
```

#### 3.1.4 Strategic Reasoning
```
Input: Negotiation history, other departments' positions, resource constraints
Process:
  1. Model other departments' priorities
  2. Predict likely responses
  3. Identify win-win opportunities
  4. Calculate optimal concession strategy
  5. Determine when to hold firm vs. compromise
Output: Negotiation strategy and next move
```

### 3.2 Shared Knowledge Base
- Resource allocation best practices
- Historical negotiation patterns
- Fairness principles
- Organizational goals and constraints
- Inter-department dependencies

### 3.3 Reasoning Patterns
- **Utilitarian**: Maximize overall organizational benefit
- **Fairness-based**: Ensure equitable distribution
- **Priority-based**: Respect strategic priorities
- **Pareto-optimal**: Find solutions where no one can be better off without making others worse
- **Coalition-building**: Form alliances with compatible departments

---

## 4. Department Chatbot Specification

### 4.1 Department Profile Structure

```python
class DepartmentProfile:
    department_id: str
    department_name: str
    resource_priorities: Dict[str, float]  # Resource type -> priority weight
    current_projects: List[Project]
    historical_usage: Dict[str, float]
    constraints: Dict[str, Any]
    dependencies: List[str]  # Other departments this depends on
    strategic_objectives: List[str]
    negotiation_style: str  # "aggressive", "collaborative", "defensive", etc.
    min_acceptable_allocation: Dict[str, float]
```

### 4.2 Chatbot State Machine

```
States:
1. IDLE: Waiting for negotiation trigger
2. ASSESSING: Analyzing needs and preparing request
3. REQUESTING: Submitting resource request with justification
4. EVALUATING: Reviewing proposals from other departments
5. NEGOTIATING: Engaging in back-and-forth discussion
6. COUNTERING: Generating counter-proposals
7. ACCEPTING: Agreeing to a proposal
8. REJECTING: Declining a proposal (may trigger new round)
9. FINALIZING: Confirming final allocation
```

### 4.3 Communication Protocol

#### 4.3.1 Message Types
- **ResourceRequest**: Initial request with justification
- **Proposal**: Specific allocation proposal
- **CounterProposal**: Modified proposal in response to another
- **Argument**: Supporting evidence or reasoning
- **Acceptance**: Agreement to a proposal
- **Rejection**: Decline with reasoning
- **Question**: Request for clarification
- **Concession**: Willingness to adjust position

#### 4.3.2 Message Structure
```python
class NegotiationMessage:
    message_id: str
    sender: str  # Department ID
    receiver: str | List[str]  # Target department(s) or "all"
    message_type: str
    timestamp: datetime
    content: Dict[str, Any]
    reasoning: str  # Explanation of the message
    metadata: Dict[str, Any]
```

---

## 5. Negotiation Protocol

### 5.1 Negotiation Phases

#### Phase 1: Initialization
1. Resource pool is announced (total available resources)
2. All departments assess their needs
3. Departments submit initial resource requests

#### Phase 2: Request Submission
1. Each department submits:
   - Resource requirements (quantities, types)
   - Justification (business case, urgency, impact)
   - Priority ranking
   - Minimum acceptable allocation

#### Phase 3: Proposal Generation
1. System or designated coordinator generates initial proposal
2. Proposal attempts to satisfy all departments within constraints
3. Proposal may use Pareto optimization or fairness algorithms

#### Phase 4: Evaluation & Discussion
1. Each department evaluates the proposal
2. Departments communicate:
   - Acceptances
   - Rejections with reasoning
   - Counter-proposals
   - Questions or clarifications

#### Phase 5: Iterative Negotiation
1. Multiple rounds of:
   - Counter-proposals
   - Arguments and justifications
   - Concessions
   - Coalition formation
2. Continue until consensus or timeout

#### Phase 6: Consensus & Validation
1. Check if all departments accept a proposal
2. Validate resource feasibility
3. Finalize allocation
4. Record decision and reasoning

### 5.2 Negotiation Rules

#### Turn-Taking
- Round-robin or priority-based
- Time limits per turn
- Maximum number of rounds

#### Argumentation Rules
- Arguments must be evidence-based
- Counter-arguments must address specific points
- Appeals to fairness, strategy, or precedent are valid
- Personal attacks or irrelevant arguments are invalid

#### Consensus Criteria
- **Unanimous**: All departments accept
- **Majority**: Threshold-based (e.g., 2/3 acceptance)
- **Weighted**: Based on department importance or resource stake
- **Mediated**: Hub makes final decision if deadlock

### 5.3 Deadlock Resolution

If negotiation reaches deadlock:
1. **Mediation**: Hub proposes compromise solution
2. **Arbitration**: Hub makes binding decision based on:
   - Strategic priorities
   - Historical fairness
   - Organizational rules
3. **Escalation**: Flag for human review
4. **Partial Agreement**: Accept partial consensus, defer remaining resources

---

## 6. Resource Management Logic

### 6.1 Resource Types

#### 6.1.1 Budget/Financial Resources
- Total budget pool
- Allocation constraints
- Spending timelines
- ROI requirements

#### 6.1.2 Human Resources
- Headcount allocations
- Skill requirements
- Time commitments
- Team composition

#### 6.1.3 Infrastructure/Equipment
- Hardware/software resources
- Shared facilities
- Equipment time slots
- Capacity limits

#### 6.1.4 Time Resources
- Project timelines
- Deadline constraints
- Priority scheduling
- Dependency management

### 6.2 Allocation Algorithms

#### 6.2.1 Fairness-Based Allocation
- Equal distribution
- Proportional to department size
- Proportional to historical usage
- Proportional to strategic importance

#### 6.2.2 Priority-Based Allocation
- Rank departments by strategic priority
- Allocate to highest priority first
- Distribute remaining resources

#### 6.2.3 Pareto-Optimal Allocation
- Find allocations where no department can be better off without making others worse
- Maximize overall utility
- Consider trade-offs

#### 6.2.4 Negotiated Allocation
- Let departments negotiate freely
- Validate final agreement
- Ensure feasibility

### 6.3 Constraint Management

#### Hard Constraints
- Total resource limits (cannot exceed)
- Minimum allocations (must provide)
- Legal/regulatory requirements
- Dependencies (must satisfy before allocation)

#### Soft Constraints
- Preferred allocations
- Historical patterns
- Strategic preferences
- Department preferences

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goals**: Set up core infrastructure
- [ ] Design base thinking logic engine
- [ ] Create department chatbot framework
- [ ] Implement basic communication layer
- [ ] Set up resource pool manager
- [ ] Create data models and schemas

**Deliverables**:
- Base logic engine module
- Department chatbot base class
- Message passing infrastructure
- Resource pool data structures

### Phase 2: Core Negotiation (Weeks 3-4)
**Goals**: Implement basic negotiation capabilities
- [ ] Implement negotiation protocol
- [ ] Create message types and handlers
- [ ] Build argument generation logic
- [ ] Implement proposal evaluation
- [ ] Create turn-taking mechanism

**Deliverables**:
- Negotiation orchestrator
- Message handling system
- Basic argumentation engine
- Proposal generation logic

### Phase 3: Department Specialization (Weeks 5-6)
**Goals**: Create department-specific chatbots
- [ ] Define department profiles
- [ ] Implement department-specific logic
- [ ] Create department knowledge bases
- [ ] Test with 2-3 departments
- [ ] Refine negotiation strategies

**Deliverables**:
- Engineering chatbot
- Marketing chatbot
- Sales chatbot (or other initial departments)
- Department profile system

### Phase 4: Advanced Features (Weeks 7-8)
**Goals**: Add sophisticated negotiation capabilities
- [ ] Implement Pareto optimization
- [ ] Add coalition formation
- [ ] Create consensus mechanisms
- [ ] Build deadlock resolution
- [ ] Add audit logging

**Deliverables**:
- Pareto optimization module
- Consensus validator
- Deadlock resolution system
- Audit trail system

### Phase 5: REST API Development (Weeks 9-10)
**Goals**: Build REST API endpoints and authentication
- [ ] Implement API key authentication system
- [ ] Create department management endpoints
- [ ] Create resource pool management endpoints
- [ ] Create negotiation session endpoints
- [ ] Implement real-time streaming (SSE/WebSocket)
- [ ] Add analytics and reporting endpoints
- [ ] Create API documentation (OpenAPI/Swagger)
- [ ] Write API integration tests

**Deliverables**:
- Complete REST API with all endpoints
- API authentication system
- Real-time communication infrastructure
- API documentation
- Integration test suite

### Phase 6: Frontend Development (Weeks 11-13)
**Goals**: Build web frontend for system interaction
- [ ] Set up frontend project (React/Vue/Next.js)
- [ ] Create API client library
- [ ] Build department management UI
- [ ] Build negotiation dashboard
- [ ] Implement real-time updates (SSE/WebSocket)
- [ ] Create resource allocation visualizations
- [ ] Add analytics dashboards
- [ ] Implement manual intervention UI
- [ ] Create reports and exports
- [ ] Responsive design and mobile support

**Deliverables**:
- Complete web frontend application
- Real-time monitoring interface
- Visualization components
- Analytics dashboards
- Mobile-responsive design

### Phase 7: Integration & Testing (Weeks 14-15)
**Goals**: Integrate all components and test end-to-end
- [ ] Integrate frontend with backend API
- [ ] Integrate with existing graph database (Neo4j)
- [ ] Connect to LLM services
- [ ] Add RAG capabilities for department knowledge
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Security audit
- [ ] Load testing

**Deliverables**:
- Fully integrated system
- End-to-end test suite
- Performance benchmarks
- Security assessment
- Complete documentation

---

## 8. Technical Requirements

### 8.1 Technology Stack

#### Backend
- **Language**: Python 3.8+
- **Framework**: FastAPI (aligns with existing system)
- **LLM Integration**: 
  - Ollama (local models)
  - Gemini API
  - OpenAI API (if needed)
- **Graph Database**: Neo4j (existing)
- **Vector Database**: ChromaDB (existing, for RAG)
- **Message Queue**: Redis or RabbitMQ (for inter-chatbot communication)

#### Frontend
- **Framework**: Gradio (existing) or React/Vue for advanced UI
- **Visualization**: D3.js, Plotly, or similar
- **Real-time Updates**: WebSockets or Server-Sent Events

#### Infrastructure
- **Containerization**: Docker (existing)
- **Orchestration**: Docker Compose (existing)
- **Monitoring**: Logging, metrics collection
- **Storage**: PostgreSQL or MongoDB for structured data (if needed beyond Neo4j)

### 8.2 Integration Points

#### Existing System Components
- **graph.py**: Extend for negotiation graph storage
- **coach.py**: Adapt for department-specific coaching
- **rag.py**: Use for department knowledge retrieval
- **pareto.py**: Extend for multi-department optimization
- **preference.py**: Use for understanding department priorities

#### New Components Needed
- `department_chatbot.py`: Base class for department chatbots
- `base_logic_engine.py`: Core reasoning engine
- `resource_manager.py`: Resource pool management
- `negotiation_orchestrator.py`: Negotiation coordination
- `consensus_validator.py`: Agreement validation
- `message_broker.py`: Inter-chatbot communication
- `api_auth.py`: API key authentication and authorization
- `api_routes.py`: REST API endpoint definitions (extends main.py)
- `websocket_manager.py`: WebSocket connection management
- `sse_manager.py`: Server-Sent Events manager

### 8.3 Data Models

#### Department
```python
{
    "department_id": str,
    "name": str,
    "profile": DepartmentProfile,
    "current_state": Dict,
    "negotiation_history": List[NegotiationMessage]
}
```

#### Resource Pool
```python
{
    "pool_id": str,
    "resource_type": str,
    "total_available": float,
    "allocated": Dict[str, float],  # department_id -> amount
    "pending_requests": List[ResourceRequest],
    "constraints": Dict
}
```

#### Negotiation Session
```python
{
    "session_id": str,
    "participants": List[str],  # department IDs
    "resource_pool": ResourcePool,
    "phases": List[NegotiationPhase],
    "messages": List[NegotiationMessage],
    "final_allocation": Dict[str, float] | None,
    "status": str,  # "active", "consensus", "deadlock", "completed"
    "start_time": datetime,
    "end_time": datetime | None
}
```

---

## 9. REST API & Frontend Integration

### 9.1 API Architecture Overview

The system will expose a RESTful API built on FastAPI (extending the existing `main.py` structure) that allows frontend applications to:
- Create and manage negotiation sessions
- Monitor real-time negotiations
- Interact with department chatbots
- View resource allocations and analytics
- Manage departments and resource pools

### 9.2 API Authentication

#### 9.2.1 API Key Authentication
- **Method**: API key passed via HTTP header
- **Header Name**: `X-API-Key` or `Authorization: Bearer <api_key>`
- **Implementation**: FastAPI dependency for API key validation
- **Key Management**: 
  - Keys stored in database (PostgreSQL/Neo4j) or environment variables
  - Key rotation support
  - Role-based access control (admin, user, read-only)

```python
# Example API key validation
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    # Validate API key against database
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

### 9.3 REST API Endpoints

#### 9.3.1 Authentication Endpoints

**POST `/api/v1/auth/register`**
- Register a new API key
- **Request Body**:
  ```json
  {
    "organization": "string",
    "email": "string",
    "role": "admin" | "user" | "readonly"
  }
  ```
- **Response**:
  ```json
  {
    "api_key": "string",
    "expires_at": "datetime",
    "role": "string"
  }
  ```

**POST `/api/v1/auth/validate`**
- Validate an API key
- **Headers**: `X-API-Key: <key>`
- **Response**:
  ```json
  {
    "valid": true,
    "role": "string",
    "expires_at": "datetime"
  }
  ```

#### 9.3.2 Department Management Endpoints

**GET `/api/v1/departments`**
- List all departments
- **Query Parameters**: `?status=active|inactive`
- **Response**:
  ```json
  {
    "departments": [
      {
        "department_id": "string",
        "name": "string",
        "status": "active",
        "current_projects": [],
        "resource_priorities": {}
      }
    ]
  }
  ```

**GET `/api/v1/departments/{department_id}`**
- Get department details
- **Response**:
  ```json
  {
    "department_id": "string",
    "name": "string",
    "profile": {},
    "current_state": {},
    "negotiation_history": []
  }
  ```

**POST `/api/v1/departments`**
- Create a new department chatbot
- **Request Body**:
  ```json
  {
    "name": "string",
    "resource_priorities": {},
    "constraints": {},
    "strategic_objectives": []
  }
  ```
- **Response**: Department object

**PUT `/api/v1/departments/{department_id}`**
- Update department profile
- **Request Body**: Partial department profile
- **Response**: Updated department object

**DELETE `/api/v1/departments/{department_id}`**
- Deactivate a department chatbot
- **Response**: `{"message": "Department deactivated"}`

#### 9.3.3 Resource Pool Management Endpoints

**GET `/api/v1/resource-pools`**
- List all resource pools
- **Response**:
  ```json
  {
    "pools": [
      {
        "pool_id": "string",
        "resource_type": "budget|personnel|equipment|time",
        "total_available": 1000000.0,
        "allocated": {},
        "available": 500000.0
      }
    ]
  }
  ```

**GET `/api/v1/resource-pools/{pool_id}`**
- Get resource pool details
- **Response**: Full resource pool object

**POST `/api/v1/resource-pools`**
- Create a new resource pool
- **Request Body**:
  ```json
  {
    "resource_type": "budget",
    "total_available": 1000000.0,
    "constraints": {},
    "description": "string"
  }
  ```
- **Response**: Resource pool object

**PUT `/api/v1/resource-pools/{pool_id}`**
- Update resource pool (e.g., adjust total available)
- **Request Body**: Partial resource pool data
- **Response**: Updated resource pool

#### 9.3.4 Negotiation Session Endpoints

**POST `/api/v1/negotiations`**
- Create a new negotiation session
- **Request Body**:
  ```json
  {
    "participants": ["dept1", "dept2", "dept3"],
    "resource_pool_id": "string",
    "negotiation_type": "budget|personnel|equipment|mixed",
    "deadline": "datetime",
    "rules": {
      "max_rounds": 10,
      "time_per_turn": 300,
      "consensus_type": "unanimous|majority|weighted"
    }
  }
  ```
- **Response**:
  ```json
  {
    "session_id": "string",
    "status": "initializing",
    "created_at": "datetime",
    "participants": []
  }
  ```

**GET `/api/v1/negotiations`**
- List all negotiation sessions
- **Query Parameters**: 
  - `?status=active|completed|deadlock`
  - `?department_id=<id>` (filter by participant)
  - `?limit=10&offset=0`
- **Response**: List of negotiation sessions

**GET `/api/v1/negotiations/{session_id}`**
- Get negotiation session details
- **Response**:
  ```json
  {
    "session_id": "string",
    "status": "active",
    "participants": [],
    "current_phase": "evaluation",
    "round": 3,
    "messages": [],
    "proposals": [],
    "final_allocation": null,
    "start_time": "datetime",
    "last_update": "datetime"
  }
  ```

**GET `/api/v1/negotiations/{session_id}/messages`**
- Get all messages in a negotiation
- **Query Parameters**: `?limit=50&offset=0`
- **Response**: List of messages

**GET `/api/v1/negotiations/{session_id}/proposals`**
- Get all proposals in a negotiation
- **Response**: List of proposals with accept/reject status

**POST `/api/v1/negotiations/{session_id}/intervene`**
- Manual intervention in negotiation (admin only)
- **Request Body**:
  ```json
  {
    "action": "propose|force_consensus|escalate",
    "proposal": {}  // if action is "propose"
  }
  ```
- **Response**: Updated negotiation status

**POST `/api/v1/negotiations/{session_id}/cancel`**
- Cancel an active negotiation
- **Response**: `{"message": "Negotiation cancelled"}`

#### 9.3.5 Real-Time Updates Endpoint

**GET `/api/v1/negotiations/{session_id}/stream`**
- Server-Sent Events (SSE) stream for real-time updates
- **Headers**: `X-API-Key: <key>`
- **Response**: SSE stream with events:
  ```json
  event: message
  data: {"type": "new_message", "message": {...}}

  event: proposal
  data: {"type": "new_proposal", "proposal": {...}}

  event: status
  data: {"type": "status_change", "status": "consensus"}
  ```

**WebSocket Alternative**: `WS /api/v1/negotiations/{session_id}/ws`
- Bidirectional WebSocket connection
- Supports sending messages and receiving updates

#### 9.3.6 Analytics & Reporting Endpoints

**GET `/api/v1/analytics/negotiations`**
- Get negotiation analytics
- **Query Parameters**: 
  - `?start_date=<date>&end_date=<date>`
  - `?department_id=<id>`
- **Response**:
  ```json
  {
    "total_negotiations": 50,
    "consensus_rate": 0.85,
    "avg_time_to_consensus": 3600,
    "avg_rounds": 4.2,
    "deadlock_rate": 0.15
  }
  ```

**GET `/api/v1/analytics/allocations`**
- Get resource allocation analytics
- **Response**: Allocation statistics and trends

**GET `/api/v1/analytics/departments/{department_id}`**
- Get department-specific analytics
- **Response**: Department performance metrics

**GET `/api/v1/reports/{report_type}`**
- Generate reports (PDF/JSON)
- **Query Parameters**: `?format=pdf|json`
- **Report Types**: `negotiation_summary`, `allocation_history`, `department_performance`

### 9.4 API Response Format

#### Standard Success Response
```json
{
  "success": true,
  "data": {},
  "message": "string",
  "timestamp": "datetime"
}
```

#### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {}
  },
  "timestamp": "datetime"
}
```

#### HTTP Status Codes
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (e.g., duplicate)
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### 9.5 Frontend Integration Architecture

#### 9.5.1 Frontend Technology Stack

**Option 1: React + TypeScript (Recommended)**
- **Framework**: React 18+
- **Language**: TypeScript
- **State Management**: Redux Toolkit or Zustand
- **HTTP Client**: Axios or Fetch API
- **Real-time**: EventSource (SSE) or Socket.io (WebSocket)
- **UI Library**: Material-UI, Ant Design, or Tailwind CSS
- **Visualization**: Recharts, D3.js, or Plotly.js
- **Build Tool**: Vite or Create React App

**Option 2: Vue.js + TypeScript**
- **Framework**: Vue 3
- **Language**: TypeScript
- **State Management**: Pinia
- **HTTP Client**: Axios
- **Real-time**: Native WebSocket or Socket.io
- **UI Library**: Vuetify or Element Plus

**Option 3: Next.js (Full-Stack)**
- **Framework**: Next.js 14+
- **Language**: TypeScript
- **API Routes**: For server-side proxy if needed
- **Real-time**: Server-Sent Events or WebSockets

#### 9.5.2 Frontend Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts          # API client configuration
│   │   ├── auth.ts            # Authentication utilities
│   │   ├── departments.ts     # Department endpoints
│   │   ├── negotiations.ts    # Negotiation endpoints
│   │   ├── resourcePools.ts   # Resource pool endpoints
│   │   └── analytics.ts       # Analytics endpoints
│   ├── components/
│   │   ├── common/            # Reusable components
│   │   ├── departments/       # Department management UI
│   │   ├── negotiations/      # Negotiation UI
│   │   ├── resourcePools/     # Resource pool UI
│   │   └── analytics/         # Analytics dashboards
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Negotiations.tsx
│   │   ├── Departments.tsx
│   │   ├── ResourcePools.tsx
│   │   └── Analytics.tsx
│   ├── store/                 # State management
│   │   ├── slices/
│   │   │   ├── negotiations.ts
│   │   │   ├── departments.ts
│   │   │   └── resourcePools.ts
│   │   └── store.ts
│   ├── hooks/
│   │   ├── useNegotiation.ts  # Custom hooks
│   │   ├── useRealtime.ts     # Real-time updates
│   │   └── useApi.ts          # API hooks
│   ├── utils/
│   │   ├── formatters.ts
│   │   └── validators.ts
│   ├── types/
│   │   └── index.ts           # TypeScript types
│   └── App.tsx
├── public/
└── package.json
```

#### 9.5.3 API Client Implementation

**Base API Client (TypeScript)**
```typescript
// src/api/client.ts
import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY || '';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api/v1`,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add API key to every request
        if (API_KEY) {
          config.headers['X-API-Key'] = API_KEY;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response.data,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          console.error('Invalid API key');
        }
        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, params?: any): Promise<T> {
    return this.client.get(url, { params });
  }

  async post<T>(url: string, data?: any): Promise<T> {
    return this.client.post(url, data);
  }

  async put<T>(url: string, data?: any): Promise<T> {
    return this.client.put(url, data);
  }

  async delete<T>(url: string): Promise<T> {
    return this.client.delete(url);
  }
}

export const apiClient = new ApiClient();
```

**Department API Module**
```typescript
// src/api/departments.ts
import { apiClient } from './client';

export interface Department {
  department_id: string;
  name: string;
  status: 'active' | 'inactive';
  resource_priorities: Record<string, number>;
  current_projects: any[];
}

export const departmentsApi = {
  list: (status?: string) => 
    apiClient.get<{ departments: Department[] }>('/departments', { status }),
  
  get: (departmentId: string) => 
    apiClient.get<Department>(`/departments/${departmentId}`),
  
  create: (data: Partial<Department>) => 
    apiClient.post<Department>('/departments', data),
  
  update: (departmentId: string, data: Partial<Department>) => 
    apiClient.put<Department>(`/departments/${departmentId}`, data),
  
  delete: (departmentId: string) => 
    apiClient.delete(`/departments/${departmentId}`),
};
```

**Negotiation API Module**
```typescript
// src/api/negotiations.ts
import { apiClient } from './client';

export interface NegotiationSession {
  session_id: string;
  status: 'active' | 'completed' | 'deadlock';
  participants: string[];
  current_phase: string;
  round: number;
  messages: any[];
  proposals: any[];
  final_allocation: Record<string, number> | null;
}

export const negotiationsApi = {
  create: (data: {
    participants: string[];
    resource_pool_id: string;
    negotiation_type: string;
    deadline?: string;
    rules?: any;
  }) => apiClient.post<NegotiationSession>('/negotiations', data),
  
  list: (params?: { status?: string; department_id?: string }) => 
    apiClient.get<{ negotiations: NegotiationSession[] }>('/negotiations', params),
  
  get: (sessionId: string) => 
    apiClient.get<NegotiationSession>(`/negotiations/${sessionId}`),
  
  getMessages: (sessionId: string, limit = 50, offset = 0) => 
    apiClient.get(`/negotiations/${sessionId}/messages`, { limit, offset }),
  
  getProposals: (sessionId: string) => 
    apiClient.get(`/negotiations/${sessionId}/proposals`),
  
  intervene: (sessionId: string, action: string, proposal?: any) => 
    apiClient.post(`/negotiations/${sessionId}/intervene`, { action, proposal }),
  
  cancel: (sessionId: string) => 
    apiClient.post(`/negotiations/${sessionId}/cancel`),
};
```

#### 9.5.4 Real-Time Updates Implementation

**Server-Sent Events (SSE) Hook**
```typescript
// src/hooks/useRealtime.ts
import { useEffect, useState, useRef } from 'react';

export function useNegotiationStream(sessionId: string) {
  const [messages, setMessages] = useState<any[]>([]);
  const [status, setStatus] = useState<string>('connecting');
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const API_KEY = process.env.REACT_APP_API_KEY || '';
    const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    const eventSource = new EventSource(
      `${API_URL}/api/v1/negotiations/${sessionId}/stream`,
      {
        headers: {
          'X-API-Key': API_KEY,
        },
      } as any // EventSource doesn't support custom headers directly
    );

    eventSource.onopen = () => {
      setStatus('connected');
    };

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'new_message') {
        setMessages((prev) => [...prev, data.message]);
      } else if (data.type === 'status_change') {
        setStatus(data.status);
      }
    };

    eventSource.onerror = () => {
      setStatus('error');
    };

    eventSourceRef.current = eventSource;

    return () => {
      eventSource.close();
    };
  }, [sessionId]);

  return { messages, status };
}
```

**WebSocket Alternative**
```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useState, useRef } from 'react';

export function useNegotiationWebSocket(sessionId: string) {
  const [messages, setMessages] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const API_KEY = process.env.REACT_APP_API_KEY || '';
    const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    
    const ws = new WebSocket(
      `${WS_URL}/api/v1/negotiations/${sessionId}/ws?api_key=${API_KEY}`
    );

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages((prev) => [...prev, data]);
    };

    ws.onerror = () => {
      setConnected(false);
    };

    ws.onclose = () => {
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const sendMessage = (message: any) => {
    if (wsRef.current && connected) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  return { messages, connected, sendMessage };
}
```

#### 9.5.5 Frontend Component Examples

**Negotiation Dashboard Component**
```typescript
// src/components/negotiations/NegotiationDashboard.tsx
import React, { useEffect, useState } from 'react';
import { negotiationsApi, NegotiationSession } from '../../api/negotiations';
import { useNegotiationStream } from '../../hooks/useRealtime';

export const NegotiationDashboard: React.FC<{ sessionId: string }> = ({ sessionId }) => {
  const [negotiation, setNegotiation] = useState<NegotiationSession | null>(null);
  const { messages, status } = useNegotiationStream(sessionId);

  useEffect(() => {
    const loadNegotiation = async () => {
      const data = await negotiationsApi.get(sessionId);
      setNegotiation(data);
    };
    loadNegotiation();
  }, [sessionId]);

  return (
    <div className="negotiation-dashboard">
      <h2>Negotiation: {sessionId}</h2>
      <div>Status: {negotiation?.status}</div>
      <div>Round: {negotiation?.round}</div>
      <div>Phase: {negotiation?.current_phase}</div>
      
      <div className="messages">
        <h3>Messages</h3>
        {messages.map((msg, idx) => (
          <div key={idx} className="message">
            <strong>{msg.sender}:</strong> {msg.content}
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 9.6 Frontend Deployment

#### 9.6.1 Environment Configuration
```bash
# .env.production
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_API_KEY=your_production_api_key
REACT_APP_WS_URL=wss://api.yourdomain.com
```

#### 9.6.2 Build & Deployment
- **Build**: `npm run build` (creates `build/` directory)
- **Static Hosting**: Deploy `build/` to:
  - Netlify
  - Vercel
  - AWS S3 + CloudFront
  - GitHub Pages
  - Any static hosting service

#### 9.6.3 CORS Configuration
Ensure FastAPI backend allows frontend origin:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development
        "https://yourdomain.com",  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 9.7 API Documentation

#### 9.7.1 OpenAPI/Swagger Documentation
FastAPI automatically generates OpenAPI documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

#### 9.7.2 API Documentation Best Practices
- Include request/response examples
- Document error codes
- Provide authentication instructions
- Include rate limiting information
- Document WebSocket/SSE protocols

### 9.8 Security Considerations

#### 9.8.1 API Key Security
- Store API keys securely (environment variables, secrets manager)
- Use HTTPS in production
- Implement key rotation
- Rate limiting per API key
- Monitor for suspicious activity

#### 9.8.2 Frontend Security
- Never expose API keys in client-side code (use environment variables)
- Implement request validation
- Use HTTPS only
- Implement CSRF protection if using cookies
- Sanitize user inputs

### 9.9 Example Integration Flow

1. **Frontend Initialization**
   ```typescript
   // App.tsx
   useEffect(() => {
     // Validate API key on app start
     authApi.validate().then(console.log);
   }, []);
   ```

2. **Create Negotiation**
   ```typescript
   const handleCreateNegotiation = async () => {
     const session = await negotiationsApi.create({
       participants: ['engineering', 'marketing', 'sales'],
       resource_pool_id: 'budget-q1-2024',
       negotiation_type: 'budget',
     });
     navigate(`/negotiations/${session.session_id}`);
   };
   ```

3. **Monitor Real-Time**
   ```typescript
   const { messages, status } = useNegotiationStream(sessionId);
   // Messages automatically update as negotiation progresses
   ```

4. **View Analytics**
   ```typescript
   const analytics = await analyticsApi.negotiations({
     start_date: '2024-01-01',
     end_date: '2024-03-31',
   });
   ```

---

## 10. Use Cases & Scenarios

### 9.1 Scenario 1: Annual Budget Allocation
**Context**: Annual planning, multiple departments competing for budget
- **Participants**: Engineering, Marketing, Sales, Operations, HR
- **Resources**: $10M total budget
- **Duration**: 2-week negotiation period
- **Expected Outcome**: Consensus on budget split with justifications

### 9.2 Scenario 2: Emergency Resource Reallocation
**Context**: Critical project needs immediate resources
- **Participants**: All departments
- **Resources**: Personnel, budget, equipment
- **Duration**: 24-hour rapid negotiation
- **Expected Outcome**: Quick consensus or escalation

### 9.3 Scenario 3: Quarterly Planning
**Context**: Quarterly resource planning
- **Participants**: All departments
- **Resources**: Multiple resource types
- **Duration**: 1-week negotiation
- **Expected Outcome**: Quarterly allocation plan

### 9.4 Scenario 4: Project-Specific Allocation
**Context**: New strategic project requires cross-department resources
- **Participants**: Relevant departments only
- **Resources**: Project-specific needs
- **Duration**: Project timeline
- **Expected Outcome**: Project resource agreement

---

## 10. Success Metrics

### 10.1 Negotiation Metrics
- **Consensus Rate**: Percentage of negotiations reaching agreement
- **Time to Consensus**: Average time to reach agreement
- **Number of Rounds**: Average negotiation rounds needed
- **Deadlock Rate**: Frequency of deadlocks requiring intervention

### 10.2 Allocation Quality Metrics
- **Satisfaction Score**: Department satisfaction with allocations
- **Fairness Index**: Measured fairness of distributions
- **Efficiency**: Resource utilization rates
- **Strategic Alignment**: Alignment with organizational goals

### 10.3 System Performance Metrics
- **Response Time**: Chatbot response latency
- **Throughput**: Negotiations processed per time period
- **Scalability**: Performance with increasing departments
- **Reliability**: System uptime and error rates

---

## 11. Risk Mitigation

### 11.1 Technical Risks
- **Risk**: Chatbots fail to reach consensus
  - **Mitigation**: Implement robust deadlock resolution, allow human override
- **Risk**: System performance degrades with many departments
  - **Mitigation**: Optimize algorithms, implement caching, use async processing
- **Risk**: LLM inconsistencies in reasoning
  - **Mitigation**: Use structured prompts, implement validation, add fallback logic

### 11.2 Business Risks
- **Risk**: Allocations don't align with business needs
  - **Mitigation**: Include business rules in base logic, allow manual review
- **Risk**: Departments game the system
  - **Mitigation**: Implement fairness checks, audit trails, validation rules
- **Risk**: Lack of transparency
  - **Mitigation**: Full audit logging, reasoning explanations, visualization

### 11.3 Adoption Risks
- **Risk**: Departments don't trust AI allocations
  - **Mitigation**: Transparent reasoning, human oversight, gradual rollout
- **Risk**: Complex system difficult to use
  - **Mitigation**: Intuitive UI, good documentation, training

---

## 12. Future Enhancements

### 12.1 Advanced Features
- **Machine Learning**: Learn from past negotiations to improve strategies
- **Predictive Analytics**: Predict resource needs and optimize proactively
- **Multi-Objective Optimization**: Optimize for multiple goals simultaneously
- **Dynamic Reallocation**: Adjust allocations based on changing conditions
- **External Factors**: Incorporate market conditions, company performance

### 12.2 Integration Opportunities
- **ERP Systems**: Integrate with enterprise resource planning
- **Project Management**: Connect with project management tools
- **Financial Systems**: Link with accounting and budgeting systems
- **HR Systems**: Integrate with human resource management

### 12.3 Advanced Negotiation
- **Multi-Party Coalitions**: Support department alliances
- **Iterative Refinement**: Continuous optimization over time
- **Stakeholder Input**: Incorporate external stakeholder preferences
- **Scenario Planning**: Test different allocation scenarios

---

## 13. Open Questions & Decisions Needed

### 13.1 Design Decisions
- [ ] How many departments to start with?
- [ ] Which resource types to prioritize?
- [ ] What consensus mechanism to use?
- [ ] How to handle partial agreements?
- [ ] What level of human oversight?

### 13.2 Technical Decisions
- [ ] Message queue technology (Redis vs RabbitMQ)?
- [ ] Database schema for negotiations?
- [ ] Real-time vs batch processing?
- [ ] Caching strategy?
- [ ] API design (REST vs GraphQL vs gRPC)?

### 13.3 Business Decisions
- [ ] Who has authority to override AI decisions?
- [ ] What are the organizational rules for allocation?
- [ ] How to handle confidential information?
- [ ] What reporting requirements?
- [ ] How to measure success?

---

## 14. Next Steps

### Immediate Actions
1. **Review and refine this plan** with stakeholders
2. **Answer open questions** in Section 13
3. **Prioritize features** for MVP
4. **Create detailed technical specifications** for Phase 1
5. **Set up development environment** and project structure

### Short-term (Next 2 Weeks)
1. Begin Phase 1 implementation
2. Create proof-of-concept with 2 departments
3. Test base thinking logic
4. Design data models
5. Set up development infrastructure

### Medium-term (Next 2 Months)
1. Complete Phases 1-3
2. Test with real scenarios
3. Gather feedback
4. Iterate on design
5. Prepare for integration

---

## Appendix A: Example Negotiation Flow

```
Round 1:
- Engineering: "We need $3M for infrastructure upgrades, critical for Q2 launch"
- Marketing: "We need $2.5M for campaign, supports revenue goals"
- Sales: "We need $2M for team expansion, directly impacts revenue"

Round 2:
- System proposes: Eng $2.5M, Marketing $2M, Sales $1.5M (total $6M available)
- Engineering: "Counter-propose: Eng $2.8M, Marketing $1.8M, Sales $1.4M"
- Marketing: "Reject, we need minimum $2M for campaign effectiveness"
- Sales: "Accept if we get $1.6M"

Round 3:
- Engineering: "Propose: Eng $2.6M, Marketing $2M, Sales $1.4M"
- Marketing: "Accept"
- Sales: "Accept"
- Engineering: "Accept"
- Consensus reached!
```

---

## Appendix B: Glossary

- **Base Thinking Logic**: Shared reasoning framework used by all department chatbots
- **Consensus**: Agreement among all participating departments
- **Deadlock**: Situation where negotiation cannot progress
- **Department Chatbot**: AI agent representing a specific department
- **Negotiation Round**: One cycle of proposal and response
- **Pareto-Optimal**: Allocation where no department can be better off without making others worse
- **Resource Pool**: Total available resources for allocation
- **Resource Request**: Department's statement of resource needs with justification

---

---

## Appendix C: Quick API Integration Guide

### C.1 Getting Started

1. **Obtain API Key**
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "organization": "My Company",
       "email": "admin@company.com",
       "role": "admin"
     }'
   ```

2. **Test API Connection**
   ```bash
   curl -X GET http://localhost:8000/api/v1/departments \
     -H "X-API-Key: your_api_key_here"
   ```

3. **Create a Negotiation**
   ```bash
   curl -X POST http://localhost:8000/api/v1/negotiations \
     -H "X-API-Key: your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{
       "participants": ["engineering", "marketing"],
       "resource_pool_id": "budget-2024",
       "negotiation_type": "budget"
     }'
   ```

### C.2 Frontend Quick Start

1. **Install Dependencies**
   ```bash
   npm install axios
   # or
   yarn add axios
   ```

2. **Set Environment Variables**
   ```bash
   # .env
   REACT_APP_API_URL=http://localhost:8000
   REACT_APP_API_KEY=your_api_key_here
   ```

3. **Use API Client**
   ```typescript
   import { departmentsApi } from './api/departments';
   
   // Fetch departments
   const departments = await departmentsApi.list();
   console.log(departments);
   ```

### C.3 Common Integration Patterns

**Pattern 1: Polling for Updates**
```typescript
useEffect(() => {
  const interval = setInterval(async () => {
    const session = await negotiationsApi.get(sessionId);
    setNegotiation(session);
  }, 5000); // Poll every 5 seconds
  
  return () => clearInterval(interval);
}, [sessionId]);
```

**Pattern 2: Real-Time with SSE**
```typescript
const { messages, status } = useNegotiationStream(sessionId);
// Automatically updates when new messages arrive
```

**Pattern 3: Error Handling**
```typescript
try {
  const result = await negotiationsApi.create(data);
} catch (error) {
  if (error.response?.status === 401) {
    // Handle invalid API key
  } else if (error.response?.status === 400) {
    // Handle validation error
  }
}
```

---

**Document Version**: 1.1  
**Last Updated**: [Current Date]  
**Status**: Draft - Ready for Review and Refinement  
**Changes**: Added REST API & Frontend Integration section (Section 9)

