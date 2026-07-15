from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class MonocleSpanType(StrEnum):
    """All possible span types in Monocle framework"""
    # Basic types
    GENERIC = "generic"                              # Default/fallback span type
    WORKFLOW = "workflow"                            # Workflow/root span that groups related operations
    
    # Agentic operations
    AGENTIC_DELEGATION = "agentic.delegation"        # Agent delegation operations
    AGENTIC_TOOL_INVOCATION = "agentic.tool.invocation"  # Agent tool invocation
    AGENTIC_INVOCATION = "agentic.invocation"        # General agent invocation
    AGENTIC_MCP_INVOCATION = "agentic.mcp.invocation"  # MCP (Model Context Protocol) invocation
    AGENTIC_REQUEST = "agentic.request"              # Agent request handling
    
    # HTTP operations
    HTTP_PROCESS = "http.process"                    # HTTP request processing
    HTTP_SEND = "http.send"                          # HTTP request sending
    
    # Inference operations
    INFERENCE = "inference"                          # Direct LLM call like OpenAI, Anthropic
    INFERENCE_FRAMEWORK = "inference.framework"     # LangChain's inference operation
    INFERENCE_MODELAPI = "inference.modelapi"       # OpenAI API call within framework context
    
    # Embedding operations
    EMBEDDING = "embedding"                          # API call to generate vector embeddings
    EMBEDDING_MODELAPI = "embedding.modelapi"       # Embedding called from within framework
    
    # Retrieval operations
    RETRIEVAL = "retrieval"                          # Vector/data retrieval operations
    RETRIEVAL_EMBEDDING = "retrieval.embedding"     # Vector datastore calling API to compute embeddings
    
    # Legacy/compatibility
    CUSTOM = "custom"                                # Custom span types

# =============================================================================
# ENTITY DEFINITIONS
# =============================================================================

class EntityType(StrEnum):
    """Entity types that can participate in spans"""
    AGENT = "agent"
    TOOL = "tool"
    TRANSFER = "transfer"
    MODEL = "model"
    SERVICE = "service"
    API = "api"

class ESpanAttribute(StrEnum):
    """Standard entity attributes that can be referenced in span validation"""
    # Core entity attributes (without numbers - will be mapped during validation)
    ENTITY_NAME = "entity.name"
    ENTITY_TYPE = "entity.type"

    ENTITY_FROM_AGENT = "entity.from_agent"
    ENTITY_TO_AGENT = "entity.to_agent"
    
    # Provider and service attributes (without numbers - will be mapped during validation)  
    ENTITY_PROVIDER_NAME = "entity.provider_name"
    ENTITY_INFERENCE_ENDPOINT = "entity.inference_endpoint"
    ENTITY_URL = "entity.url"
    ENTITY_DESCRIPTION = "entity.description"
    ENTITY_DEPLOYMENT = "entity.deployment"
    
    # Additional attributes
    ENTITY_COUNT = "entity.count"
    
    # Core span attributes
    SPAN_TYPE = "span.type"
    WORKFLOW_NAME = "workflow.name"
    
    # Monocle core attributes
    MONOCLE_VERSION = "monocle_apptrace.version"
    MONOCLE_LANGUAGE = "monocle_apptrace.language"
    SPAN_SOURCE = "span_source"
    
    # Scope attributes
    SCOPE_AGENTIC_REQUEST = "scope.agentic.request"
    SCOPE_AGENTIC_INVOCATION = "scope.agentic.invocation"
    
    # HTTP attributes
    HTTP_METHOD = "http.method"
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_URL = "http.url"
    HTTP_REQUEST_SIZE = "http.request_size"
    HTTP_RESPONSE_SIZE = "http.response_size"
    HTTP_USER_AGENT = "http.user_agent"
    HTTP_TIMEOUT = "http.timeout"
    
    # Additional attributes
    SERVER_NAME = "server.name"
    API_ENDPOINT = "api.endpoint"
    API_VERSION = "api.version"
    FRAMEWORK_VERSION = "framework.version"
    CHAIN_TYPE = "chain.type"
    
    # Tool attributes
    TOOL_PARAMETERS = "tool.parameters"
    
    # Entity property attributes (for SimpleEntity.properties)
    WORKFLOW_NAME_PROP = "workflow_name"
    WORKFLOW_TYPE_PROP = "workflow_type"
    SERVICE_NAME_PROP = "service_name"
    SERVICE_TYPE_PROP = "service_type"
    TOOL_NAME_PROP = "tool_name"
    DESCRIPTION_PROP = "description"
    AGENT_NAME_PROP = "agent_name"
    URL_PROP = "url"
    PROVIDER_NAME_PROP = "provider_name"
    INFERENCE_ENDPOINT_PROP = "inference_endpoint"
    MODEL_NAME_PROP = "model_name"

class Props:
    """Property accessor for entity attributes with dot notation support"""
    def __init__(self, attributes: Optional[set[ESpanAttribute]] = None):
        if attributes:
            # Store by enum name for dot notation access
            self._attributes = {attr.name.split('.')[-1].upper(): attr for attr in attributes}
            self._original_set = attributes
        else:
            self._attributes = {}
            self._original_set = set()

    def __getattr__(self, item):
        if item in self._attributes:
            return self._attributes[item]
        raise AttributeError(f"{item} not found in properties")
    
    def __iter__(self):
        """Make Props iterable"""
        return iter(self._original_set)
    
    def __len__(self):
        """Return number of properties"""
        return len(self._original_set)
    
    def __bool__(self):
        """Return True if there are properties"""
        return bool(self._original_set)
    

class Entity(BaseModel):
    """Simplified entity definition"""
    type: EntityType = Field(..., description="Type of entity")
    name: str = Field(..., description="Name of the entity")
    props: Optional[set[ESpanAttribute]] = Field(None, description="Set of entity attributes that this entity exposes")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create props accessor after initialization
        self.props = Props(self.props)
    
    def get_property_strings(self) -> List[str]:
        """Get entity properties as a list of strings"""
        if not self.props:
            return []
        return [str(prop) for prop in self.props]
    
    def has_property(self, attribute: ESpanAttribute) -> bool:
        """Check if entity has a specific property"""
        return self.props is not None and attribute in self.props
    
    def get_properties_dict(self) -> dict[str, str]:
        """Get entity properties as a dictionary mapping enum names to string values"""
        if not self.props:
            return {}
        return {prop.name: str(prop) for prop in self.props}

# =============================================================================
# SPAN ATTRIBUTES AND ENTITY ATTRIBUTES
# =============================================================================

class SpanAttributes(BaseModel):
    """Attributes that should be checked for each span"""
    required_attributes: List[str] = Field(default_factory=list, description="Attributes that must be present")
    optional_attributes: List[str] = Field(default_factory=list, description="Attributes that may be present") 
    forbidden_attributes: List[str] = Field(default_factory=list, description="Attributes that must not be present")
    
    # Validation rules
    min_duration_ms: Optional[float] = Field(None, description="Minimum span duration in milliseconds")
    max_duration_ms: Optional[float] = Field(None, description="Maximum span duration in milliseconds")
    must_have_parent: bool = Field(False, description="Whether span must have a parent")
    must_be_root: bool = Field(False, description="Whether span must be root (no parent)")

class EventSchema(BaseModel):
    """Schema for span events"""
    name: str = Field(..., description="Event name")
    required_attributes: List[str] = Field(default_factory=list, description="Required event attributes")
    optional_attributes: List[str] = Field(default_factory=list, description="Optional event attributes")

# =============================================================================
# SPAN SCHEMA DEFINITION
# =============================================================================

class MonocleSpanSchema(BaseModel):
    """Simplified span schema for testing"""
    span_type: MonocleSpanType = Field(..., description="Type of span")
    entities: Optional[List[Entity]] = Field(None, description="Entities involved in this span")
    attributes: SpanAttributes = Field(default_factory=SpanAttributes, description="Attributes to validate")
    events: Optional[List[EventSchema]] = Field(None, description="Expected events in the span")
    
    # Parent-child relationship rules
    allowed_ancestors: Optional[List[MonocleSpanType]] = Field(None, description="Allowed ancestor span types in the parental path")
    allowed_descendants: Optional[List[MonocleSpanType]] = Field(None, description="Allowed descendant span types")
    
    # Test configuration
    expect_success: bool = Field(True, description="Whether span should complete successfully")
    expect_errors: bool = Field(False, description="Whether errors are expected")
    expect_warnings: bool = Field(False, description="Whether warnings are expected")

# =============================================================================
# PREDEFINED SPAN SCHEMAS
# =============================================================================

class MonocleSpanSchemaRegistry:
    """Registry of all Monocle span schemas for testing"""
    
    @staticmethod
    def get_entity_property_strings(entity: Entity) -> List[str]:
        """Helper method to get entity properties as strings"""
        return entity.get_property_strings()
    
    @staticmethod
    def create_entity_with_properties(entity_type: EntityType, name: str, properties: set[ESpanAttribute]) -> Entity:
        """Helper method to create an entity and provide easy access to its properties as strings"""
        entity = Entity(type=entity_type, name=name, props=properties)
        return entity
    
    @staticmethod
    def map_entity_attributes_to_numbered(entities: List[Entity]) -> List[str]:
        """Map entity attributes to their numbered equivalents for validation
        
        Args:
            entities: List of entities in the schema
            
        Returns:
            List of numbered attribute strings (e.g., "entity.1.name", "entity.2.type")
        """
        numbered_attributes = []
        
        for i, entity in enumerate(entities, 1):
            if entity.props and hasattr(entity.props, '_attributes'):
                for attr in entity.props._attributes.values():
                    # Convert "entity.name" to "entity.1.name", "entity.type" to "entity.1.type", etc.
                    numbered_attr = attr.value.replace("entity.", f"entity.{i}.")
                    numbered_attributes.append(numbered_attr)
        
        return numbered_attributes
    
    @staticmethod
    def build_attributes_from_entities(entities: List[Entity], required_props: set[ESpanAttribute] = None, optional_props: set[ESpanAttribute] = None) -> SpanAttributes:
        """Build SpanAttributes from entities with automatic numbering
        
        Args:
            entities: List of entities in the schema
            required_props: Set of entity attributes that should be required
            optional_props: Set of entity attributes that should be optional
            
        Returns:
            SpanAttributes with properly numbered entity attributes
        """
        required_attrs = [ESpanAttribute.SPAN_TYPE]  # Always include span.type
        optional_attrs = []
        
        for i, entity in enumerate(entities, 1):
            if entity.props and hasattr(entity.props, '_attributes'):
                for attr in entity.props._attributes.values():
                    # Convert "entity.name" to "entity.1.name", etc.
                    numbered_attr = attr.value.replace("entity.", f"entity.{i}.")
                    
                    # Determine if this should be required or optional
                    if required_props and attr in required_props:
                        required_attrs.append(numbered_attr)
                    elif optional_props and attr in optional_props:
                        optional_attrs.append(numbered_attr)
                    else:
                        # Default: make all entity attributes required
                        required_attrs.append(numbered_attr)
        
        return SpanAttributes(
            required_attributes=required_attrs,
            optional_attributes=optional_attrs
        )

    
    @staticmethod
    def generic_schema() -> MonocleSpanSchema:
        """Schema for generic/fallback spans"""
        entities = [
            Entity(type=EntityType.SERVICE, name="generic_service",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.GENERIC,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[ESpanAttribute.SPAN_TYPE],  # span.type="generic"
                optional_attributes=[
                    "operation.name",                        # operation.name="custom_operation"
                    "operation.duration",                    # operation.duration="150"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="service"
                    ESpanAttribute.ENTITY_COUNT              # entity.count="1"
                ]
            ),
            allowed_ancestors=[MonocleSpanType.WORKFLOW],
            allowed_descendants=[]
        )
    
    @staticmethod
    def workflow_schema() -> MonocleSpanSchema:
        """Schema for workflow spans (typically root spans)"""
        entities = [
            # entity.1 - Main workflow (e.g., "langchain_agent_1")
            Entity(type=EntityType.SERVICE, name="workflow_main",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.WORKFLOW_NAME}),
            # entity.2 - Hosting environment (optional)
            Entity(type=EntityType.SERVICE, name="hosting_service", 
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.WORKFLOW,
            entities=entities,
            attributes=MonocleSpanSchemaRegistry.build_attributes_from_entities(
                entities,
                required_props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.WORKFLOW_NAME},
                optional_props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE}  # For entity.2
            ),
            allowed_ancestors=[],  # Root span - no ancestors allowed
            allowed_descendants=[
                MonocleSpanType.AGENTIC_INVOCATION,
                MonocleSpanType.AGENTIC_REQUEST,
                MonocleSpanType.HTTP_PROCESS,
                MonocleSpanType.HTTP_SEND, 
                MonocleSpanType.INFERENCE,
                MonocleSpanType.INFERENCE_FRAMEWORK,
                MonocleSpanType.EMBEDDING,
                MonocleSpanType.RETRIEVAL,
                MonocleSpanType.CUSTOM
            ]
        )
    
    @staticmethod
    def agentic_delegation_schema() -> MonocleSpanSchema:
        """Schema for agent delegation spans"""
        entities = [
            Entity(type=EntityType.TRANSFER, name="delegation",
                       props={ESpanAttribute.ENTITY_TYPE, 
                              ESpanAttribute.ENTITY_FROM_AGENT, ESpanAttribute.ENTITY_TO_AGENT}),
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.AGENTIC_DELEGATION,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="agent.langgraph"
                    entities[0].props.FROM_AGENT,            # entity.1.from_agent="supervisor"
                    entities[0].props.TO_AGENT,              # entity.1.to_agent="flight_assistant"
                    ESpanAttribute.SPAN_TYPE                 # span.type="agentic.delegation"
                ]
            ),
            allowed_ancestors=[MonocleSpanType.WORKFLOW, MonocleSpanType.AGENTIC_INVOCATION],
            allowed_descendants=[]
        )
    
    @staticmethod
    def agentic_tool_invocation_schema() -> MonocleSpanSchema:
        """Schema for agent tool invocation spans"""
        entities = [
            # entity.1 - Tool being invoked (e.g., "get_weather")
            Entity(type=EntityType.TOOL, name="invoked_tool",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_DESCRIPTION}),
            # entity.2 - Agent invoking the tool (e.g., "supervisor")  
            Entity(type=EntityType.AGENT, name="invoking_agent",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.AGENTIC_TOOL_INVOCATION,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    # Will be mapped to entity.1.name, entity.1.type during validation
                    entities[0].props.ENTITY_NAME,          # entity.1.name="get_weather"
                    entities[0].props.ENTITY_TYPE,          # entity.1.type="tool"  
                    ESpanAttribute.SPAN_TYPE                 # span.type="agentic.tool.invocation"
                ],
                optional_attributes=[
                    entities[0].props.ENTITY_DESCRIPTION,   # entity.1.description="Get current weather for a location"
                    entities[1].props.ENTITY_NAME,          # entity.2.name="supervisor_agent"
                    entities[1].props.ENTITY_TYPE,          # entity.2.type="agent"
                    ESpanAttribute.TOOL_PARAMETERS           # tool.parameters="{\"location\": \"San Francisco\"}"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output", 
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.AGENTIC_INVOCATION],
            allowed_descendants=[MonocleSpanType.HTTP_SEND, MonocleSpanType.INFERENCE]
        )
    
    @staticmethod
    def agentic_invocation_schema() -> MonocleSpanSchema:
        """Schema for general agent invocation spans"""
        entities = [
            Entity(type=EntityType.AGENT, name="executing_agent",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_DESCRIPTION})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.AGENTIC_INVOCATION,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="supervisor_agent"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="agent"
                    ESpanAttribute.SPAN_TYPE                 # span.type="agentic.invocation"
                ],
                optional_attributes=[
                    entities[0].props.ENTITY_DESCRIPTION,    # entity.1.description="Main orchestrating agent"
                    "agent.instruction"                      # agent.instruction="Process user query and coordinate tasks"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.WORKFLOW, MonocleSpanType.AGENTIC_DELEGATION, MonocleSpanType.AGENTIC_REQUEST],
            allowed_descendants=[
                MonocleSpanType.AGENTIC_TOOL_INVOCATION, 
                MonocleSpanType.AGENTIC_DELEGATION,
                MonocleSpanType.AGENTIC_MCP_INVOCATION,
                MonocleSpanType.INFERENCE,
                MonocleSpanType.INFERENCE_FRAMEWORK
            ]
        )
    
    @staticmethod
    def agentic_mcp_invocation_schema() -> MonocleSpanSchema:
        """Schema for MCP (Model Context Protocol) invocation spans"""
        # Create entity first so we can reference its properties
        mcp_entity = Entity(
            type=EntityType.SERVICE, 
            name="mcp_service",
            props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_URL}
        )
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.AGENTIC_MCP_INVOCATION,
            entities=[mcp_entity],
            attributes=SpanAttributes(
                required_attributes=[
                    mcp_entity.props.ENTITY_NAME,           # entity.1.name="filesystem_mcp"
                    mcp_entity.props.ENTITY_TYPE,           # entity.1.type="service"
                    ESpanAttribute.SPAN_TYPE                 # span.type="agentic.mcp.invocation"
                ],
                optional_attributes=[
                    mcp_entity.props.ENTITY_URL,            # entity.1.url="mcp://localhost:3000/filesystem"
                    "mcp.protocol_version",                  # mcp.protocol_version="1.0"
                    "mcp.method"                             # mcp.method="read_file"
                ]
            ),
            events=[
                EventSchema(
                    name="mcp.request",
                    required_attributes=["method", "params"]
                ),
                EventSchema(
                    name="mcp.response",
                    required_attributes=["result"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.AGENTIC_INVOCATION],
            allowed_descendants=[]
        )
    
    @staticmethod
    def agentic_request_schema() -> MonocleSpanSchema:
        """Schema for agent request handling spans"""
        entities = [
            Entity(type=EntityType.AGENT, name="request_handler",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.AGENTIC_REQUEST,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="request_handler_agent"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="agent"
                    ESpanAttribute.SPAN_TYPE                 # span.type="agentic.request"
                ],
                optional_attributes=[
                    "request.id",                            # request.id="req_abc123"
                    "request.priority"                       # request.priority="high"
                ]
            ),
            events=[
                EventSchema(
                    name="request.received",
                    required_attributes=["request"]
                ),
                EventSchema(
                    name="request.processed",
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.WORKFLOW],
            allowed_descendants=[MonocleSpanType.AGENTIC_INVOCATION]
        )
    
    @staticmethod
    def http_process_schema() -> MonocleSpanSchema:
        """Schema for HTTP request processing spans (server-side)"""
        entities = [
            Entity(type=EntityType.API, name="http_server",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.HTTP_PROCESS,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    "http.method",                           # http.method="POST"
                    "http.status_code",                      # http.status_code="200"
                    "http.url",                              # http.url="/api/v1/chat"
                    ESpanAttribute.SPAN_TYPE                 # span.type="http.process"
                ],
                optional_attributes=[
                    "http.request_size",                     # http.request_size="1024"
                    "http.response_size",                    # http.response_size="2048"
                    "http.user_agent",                       # http.user_agent="LangChain/1.0"
                    "server.name"                            # server.name="api-server-1"
                ]
            ),
            events=[
                EventSchema(
                    name="http.request",
                    required_attributes=["method", "url"]
                ),
                EventSchema(
                    name="http.response", 
                    required_attributes=["status_code"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.WORKFLOW],
            allowed_descendants=[
                MonocleSpanType.AGENTIC_INVOCATION,
                MonocleSpanType.INFERENCE,
                MonocleSpanType.RETRIEVAL
            ]
        )
    
    @staticmethod
    def http_send_schema() -> MonocleSpanSchema:
        """Schema for HTTP request sending spans (client-side)"""
        entities = [
            Entity(type=EntityType.API, name="http_client",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.HTTP_SEND,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    "http.method",                           # http.method="POST"
                    "http.status_code",                      # http.status_code="200"
                    "http.url",                              # http.url="https://api.openai.com/v1/chat/completions"
                    ESpanAttribute.SPAN_TYPE                 # span.type="http.send"
                ],
                optional_attributes=[
                    "http.request_size",                     # http.request_size="1024"
                    "http.response_size",                    # http.response_size="2048"
                    "http.timeout"                           # http.timeout="30"
                ]
            ),
            events=[
                EventSchema(
                    name="http.request",
                    required_attributes=["method", "url"]
                ),
                EventSchema(
                    name="http.response", 
                    required_attributes=["status_code"]
                )
            ],
            allowed_ancestors=[
                MonocleSpanType.AGENTIC_INVOCATION,
                MonocleSpanType.AGENTIC_TOOL_INVOCATION,
                MonocleSpanType.WORKFLOW
            ],
            allowed_descendants=[]
        )
    
    @staticmethod
    def inference_schema() -> MonocleSpanSchema:
        """Schema for direct LLM inference spans (OpenAI, Anthropic)"""
        entities = [
            # entity.1 - Inference provider (e.g., "inference.openai")
            Entity(type=EntityType.SERVICE, name="inference_provider", 
                       props={ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_PROVIDER_NAME, ESpanAttribute.ENTITY_INFERENCE_ENDPOINT}),
            # entity.2 - LLM model (e.g., "gpt-4o")  
            Entity(type=EntityType.MODEL, name="llm_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.INFERENCE,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="service"
                    entities[0].props.ENTITY_PROVIDER_NAME,  # entity.1.provider_name="openai"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="gpt-4o"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    ESpanAttribute.SPAN_TYPE                 # span.type="inference"
                ],
                optional_attributes=[
                    entities[0].props.ENTITY_INFERENCE_ENDPOINT,  # entity.1.inference_endpoint="https://api.openai.com/v1/chat/completions"
                    ESpanAttribute.ENTITY_DEPLOYMENT,            # entity.deployment="azure-east-us"
                    ESpanAttribute.SCOPE_AGENTIC_REQUEST,         # scope.agentic.request="user_query_123"
                    ESpanAttribute.SCOPE_AGENTIC_INVOCATION,      # scope.agentic.invocation="langchain_agent_1"
                    ESpanAttribute.ENTITY_COUNT                   # entity.count="2"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                ),
                EventSchema(
                    name="metadata",
                    required_attributes=["completion_tokens", "prompt_tokens", "total_tokens"]
                )
            ],
            allowed_ancestors=[
                MonocleSpanType.AGENTIC_INVOCATION, 
                MonocleSpanType.AGENTIC_TOOL_INVOCATION,
                MonocleSpanType.WORKFLOW
            ],
            allowed_descendants=[]
        )
    
    @staticmethod
    def inference_framework_schema() -> MonocleSpanSchema:
        """Schema for framework inference spans (LangChain)"""
        entities = [
            # entity.1 - Framework inference provider
            Entity(type=EntityType.SERVICE, name="inference_framework",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_PROVIDER_NAME, ESpanAttribute.ENTITY_INFERENCE_ENDPOINT}),
            # entity.2 - LLM model  
            Entity(type=EntityType.MODEL, name="llm_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.INFERENCE_FRAMEWORK,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="langchain_llm"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="service"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="gpt-4o"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    ESpanAttribute.SPAN_TYPE                 # span.type="inference.framework"
                ],
                optional_attributes=[
                    entities[0].props.ENTITY_PROVIDER_NAME,      # entity.1.provider_name="openai"
                    entities[0].props.ENTITY_INFERENCE_ENDPOINT, # entity.1.inference_endpoint="https://api.openai.com/v1/chat/completions"
                    ESpanAttribute.FRAMEWORK_VERSION,            # framework.version="0.2.16"
                    ESpanAttribute.CHAIN_TYPE                     # chain.type="llm_chain"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                ),
                EventSchema(
                    name="metadata",
                    required_attributes=["completion_tokens", "prompt_tokens", "total_tokens"]
                )
            ],
            allowed_ancestors=[
                MonocleSpanType.AGENTIC_INVOCATION,
                MonocleSpanType.WORKFLOW
            ],
            allowed_descendants=[MonocleSpanType.INFERENCE_MODELAPI]
        )
    

    
    @staticmethod
    def inference_modelapi_schema() -> MonocleSpanSchema:
        """Schema for model API calls within framework context"""
        entities = [
            # entity.1 - Model API service (minimal attributes in real traces)
            Entity(type=EntityType.API, name="model_api",
                       props={ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_PROVIDER_NAME}),
            # entity.2 - LLM model (often not present in modelapi spans)
            Entity(type=EntityType.MODEL, name="llm_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.INFERENCE_MODELAPI,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    ESpanAttribute.SPAN_TYPE                 # span.type="inference.modelapi"
                ],
                optional_attributes=[
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="api"
                    entities[0].props.ENTITY_PROVIDER_NAME,  # entity.1.provider_name="openai"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="gpt-4o"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    "api.endpoint",                          # api.endpoint="/v1/chat/completions"
                    "api.version"                            # api.version="v1"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                ),
                EventSchema(
                    name="metadata",
                    required_attributes=["completion_tokens", "prompt_tokens", "total_tokens"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.INFERENCE_FRAMEWORK],
            allowed_descendants=[]
        )
    
    @staticmethod
    def embedding_schema() -> MonocleSpanSchema:
        """Schema for embedding generation spans"""
        entities = [
            Entity(type=EntityType.SERVICE, name="embedding_provider",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE}),
            Entity(type=EntityType.MODEL, name="embedding_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.EMBEDDING,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="openai_embeddings"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="service"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="text-embedding-ada-002"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    ESpanAttribute.SPAN_TYPE                 # span.type="embedding"
                ],
                optional_attributes=[
                    "embedding.dimensions",                  # embedding.dimensions="1536"
                    "embedding.model_version"                # embedding.model_version="2"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[
                MonocleSpanType.WORKFLOW,
                MonocleSpanType.RETRIEVAL,
                MonocleSpanType.RETRIEVAL_EMBEDDING
            ],
            allowed_descendants=[]
        )
    
    @staticmethod
    def embedding_modelapi_schema() -> MonocleSpanSchema:
        """Schema for embedding API calls within framework context"""
        entities = [
            Entity(type=EntityType.API, name="embedding_api",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE}),
            Entity(type=EntityType.MODEL, name="embedding_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.EMBEDDING_MODELAPI,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="openai_embedding_api"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="api"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="text-embedding-ada-002"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    ESpanAttribute.SPAN_TYPE                 # span.type="embedding.modelapi"
                ],
                optional_attributes=[
                    "api.endpoint",                          # api.endpoint="/v1/embeddings"
                    "framework.name"                         # framework.name="langchain"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[
                MonocleSpanType.RETRIEVAL,
                MonocleSpanType.RETRIEVAL_EMBEDDING
            ],
            allowed_descendants=[]
        )
    
    @staticmethod
    def retrieval_schema() -> MonocleSpanSchema:
        """Schema for vector/data retrieval spans"""
        entities = [
            Entity(type=EntityType.SERVICE, name="vector_store",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE, ESpanAttribute.ENTITY_DEPLOYMENT}),
            Entity(type=EntityType.MODEL, name="embedding_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.RETRIEVAL,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="chroma_vectorstore"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="service"
                    ESpanAttribute.SPAN_TYPE                 # span.type="retrieval"
                ],
                optional_attributes=[
                    entities[0].props.ENTITY_DEPLOYMENT,     # entity.1.deployment="local"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="text-embedding-ada-002"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    "search.query",                          # search.query="What is machine learning?"
                    "search.results_count"                   # search.results_count="5"
                ]
            ),
            events=[
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[
                MonocleSpanType.WORKFLOW,
                MonocleSpanType.AGENTIC_INVOCATION,
                MonocleSpanType.HTTP_PROCESS
            ],
            allowed_descendants=[
                MonocleSpanType.RETRIEVAL_EMBEDDING,
                MonocleSpanType.EMBEDDING,
                MonocleSpanType.EMBEDDING_MODELAPI
            ]
        )
    
    @staticmethod
    def retrieval_embedding_schema() -> MonocleSpanSchema:
        """Schema for retrieval with embedding computation spans"""
        entities = [
            Entity(type=EntityType.SERVICE, name="vector_store",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE}),
            Entity(type=EntityType.MODEL, name="embedding_model",
                       props={ESpanAttribute.ENTITY_NAME, ESpanAttribute.ENTITY_TYPE})
        ]
        
        return MonocleSpanSchema(
            span_type=MonocleSpanType.RETRIEVAL_EMBEDDING,
            entities=entities,
            attributes=SpanAttributes(
                required_attributes=[
                    entities[0].props.ENTITY_NAME,           # entity.1.name="pinecone_vectorstore"
                    entities[0].props.ENTITY_TYPE,           # entity.1.type="service"
                    entities[1].props.ENTITY_NAME,           # entity.2.name="text-embedding-ada-002"
                    entities[1].props.ENTITY_TYPE,           # entity.2.type="model"
                    ESpanAttribute.SPAN_TYPE                 # span.type="retrieval.embedding"
                ],
                optional_attributes=[
                    "search.query",                          # search.query="What is machine learning?"
                    "embedding.computed_on_demand"           # embedding.computed_on_demand="true"
                ]
            ),
            events=[
                EventSchema(
                    name="embedding.compute",
                    required_attributes=["text", "model"]
                ),
                EventSchema(
                    name="data.input",
                    required_attributes=["input"]
                ),
                EventSchema(
                    name="data.output",
                    required_attributes=["response"]
                )
            ],
            allowed_ancestors=[MonocleSpanType.RETRIEVAL],
            allowed_descendants=[
                MonocleSpanType.EMBEDDING,
                MonocleSpanType.EMBEDDING_MODELAPI
            ]
        )
    
# =============================================================================
# SCHEMA VALIDATION UTILITIES
# =============================================================================

schema_map = {
    MonocleSpanType.GENERIC: MonocleSpanSchemaRegistry.generic_schema,
    MonocleSpanType.WORKFLOW: MonocleSpanSchemaRegistry.workflow_schema,
    MonocleSpanType.AGENTIC_DELEGATION: MonocleSpanSchemaRegistry.agentic_delegation_schema,
    MonocleSpanType.AGENTIC_TOOL_INVOCATION: MonocleSpanSchemaRegistry.agentic_tool_invocation_schema,
    MonocleSpanType.AGENTIC_INVOCATION: MonocleSpanSchemaRegistry.agentic_invocation_schema,
    MonocleSpanType.AGENTIC_MCP_INVOCATION: MonocleSpanSchemaRegistry.agentic_mcp_invocation_schema,
    MonocleSpanType.AGENTIC_REQUEST: MonocleSpanSchemaRegistry.agentic_request_schema,
    MonocleSpanType.HTTP_PROCESS: MonocleSpanSchemaRegistry.http_process_schema,
    MonocleSpanType.HTTP_SEND: MonocleSpanSchemaRegistry.http_send_schema,
    MonocleSpanType.INFERENCE: MonocleSpanSchemaRegistry.inference_schema,
    MonocleSpanType.INFERENCE_FRAMEWORK: MonocleSpanSchemaRegistry.inference_framework_schema,
    MonocleSpanType.INFERENCE_MODELAPI: MonocleSpanSchemaRegistry.inference_modelapi_schema,
    MonocleSpanType.EMBEDDING: MonocleSpanSchemaRegistry.embedding_schema,
    MonocleSpanType.EMBEDDING_MODELAPI: MonocleSpanSchemaRegistry.embedding_modelapi_schema,
    MonocleSpanType.RETRIEVAL: MonocleSpanSchemaRegistry.retrieval_schema,
    MonocleSpanType.RETRIEVAL_EMBEDDING: MonocleSpanSchemaRegistry.retrieval_embedding_schema,
}


class MonocleSchemaValidator:
    """Utility class for validating spans against schemas"""
    
    @staticmethod
    def _resolve_entity_attributes(schema: MonocleSpanSchema) -> tuple[List[str], List[str]]:
        """Resolve entity attributes to their numbered equivalents
        
        Args:
            schema: The schema containing entities and attributes
            
        Returns:
            Tuple of (resolved_required_attributes, resolved_optional_attributes)
        """
        resolved_required = []
        resolved_optional = []
        
        # Process required attributes
        for attr in schema.attributes.required_attributes:
            if isinstance(attr, ESpanAttribute) and attr.value.startswith("entity."):
                # This is an entity attribute that needs numbering resolution
                # For now, we'll need to determine which entity it belongs to
                # This could be enhanced with more sophisticated mapping
                resolved_required.append(attr)
            else:
                resolved_required.append(attr)
        
        # Process optional attributes  
        for attr in schema.attributes.optional_attributes:
            if isinstance(attr, ESpanAttribute) and attr.value.startswith("entity."):
                resolved_optional.append(attr)
            else:
                resolved_optional.append(attr)
                
        return resolved_required, resolved_optional
    
    @staticmethod
    def validate_span_attributes(span, schema: MonocleSpanSchema) -> List[str]:
        """Validate span attributes against schema requirements with entity numbering"""
        errors = []
        span_attrs = getattr(span, 'attributes', {})
        
        # Resolve entity attributes to their numbered equivalents
        resolved_required, resolved_optional = MonocleSchemaValidator._resolve_entity_attributes(schema)
        
        # Check required attributes
        for attr in resolved_required:
            attr_str = str(attr)
            # Convert entity.name to entity.1.name, entity.2.name etc. based on entities
            if attr_str.startswith("entity.") and not attr_str[7].isdigit():
                # This is an unnumbered entity attribute - we need to check all possible numbered versions
                found = False
                if schema.entities:
                    for i in range(1, len(schema.entities) + 1):
                        numbered_attr = attr_str.replace("entity.", f"entity.{i}.")
                        if numbered_attr in span_attrs:
                            found = True
                            break
                if not found:
                    errors.append(f"Missing required attribute: {attr_str} (checked as entity.1.*, entity.2.*, etc.)")
            else:
                if attr_str not in span_attrs:
                    errors.append(f"Missing required attribute: {attr_str}")
        
        # Check forbidden attributes
        for attr in schema.attributes.forbidden_attributes:
            attr_str = str(attr)
            if attr_str in span_attrs:
                errors.append(f"Forbidden attribute present: {attr_str}")
        
        return errors
    
    @staticmethod
    def validate_span_events(span, schema: MonocleSpanSchema) -> List[str]:
        """Validate span events against schema requirements"""
        errors = []
        span_events = getattr(span, 'events', [])
        
        if schema.events:
            for expected_event in schema.events:
                matching_events = [e for e in span_events if e.name == expected_event.name]
                if not matching_events:
                    errors.append(f"Missing required event: {expected_event.name}")
                else:
                    # Check event attributes
                    event = matching_events[0]
                    event_attrs = getattr(event, 'attributes', {})
                    for attr in expected_event.required_attributes:
                        if attr not in event_attrs:
                            errors.append(f"Missing required attribute '{attr}' in event '{expected_event.name}'")
        
        return errors
    
