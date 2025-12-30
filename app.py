import os
import logging
import streamlit as st
import requests
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, MCPTool

# ============================================
# 1. 환경 설정 및 로깅
# ============================================
# 로컬: .env 파일 로드 / Azure: Application Settings 자동 적용
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# 2. Streamlit 페이지 설정
# ============================================
st.set_page_config(page_title="Paris Tour AI Agent", layout="wide")
st.title("🇫🇷 Paris Tour AI Assistant")
st.caption("Azure AI Foundry Agent + MCP Knowledge Base")

# ============================================
# 3. 환경 변수 로드 (필수 검증)
# ============================================
def get_required_env(key: str) -> str:
    """필수 환경 변수를 가져오고, 없으면 에러 표시"""
    value = os.environ.get(key)
    if not value:
        st.error(f"❌ 필수 환경 변수가 설정되지 않았습니다: {key}")
        st.stop()
    return value

# 환경 변수 로드
try:
    AZURE_SEARCH_ENDPOINT = get_required_env("AZURE_SEARCH_ENDPOINT")
    KB_NAME = get_required_env("AZURE_SEARCH_KB_NAME")
    PROJECT_ENDPOINT = get_required_env("PROJECT_ENDPOINT")
    PROJECT_RESOURCE_ID = get_required_env("PROJECT_RESOURCE_ID")
    PROJECT_CONNECTION_NAME = get_required_env("PROJECT_CONNECTION_NAME")
    AGENT_NAME = get_required_env("AGENT_NAME")
    AGENT_MODEL = get_required_env("AGENT_MODEL")
    MCP_ENDPOINT = f"{AZURE_SEARCH_ENDPOINT}/knowledgebases/{KB_NAME}/mcp?api-version=2025-11-01-preview"
except Exception as e:
    logger.error(f"환경 변수 로드 실패: {e}")
    st.error(f"환경 변수 설정 오류: {e}")
    st.stop()

# ============================================
# 4. 백엔드 리소스 생성 (캐싱 처리)
# ============================================
@st.cache_resource
def initialize_agent():
    """에이전트 초기화 - Web App Managed Identity 사용"""
    try:
        # DefaultAzureCredential: 로컬에서는 Azure CLI, Web App에서는 Managed Identity 사용
        credential = DefaultAzureCredential()
        logger.info("Azure 인증 성공")
        
        # [A] Project Connection 생성/업데이트
        bearer_token_provider = get_bearer_token_provider(
            credential, 
            "https://management.azure.com/.default"
        )
        headers = {"Authorization": f"Bearer {bearer_token_provider()}"}
        body = {
            "name": PROJECT_CONNECTION_NAME,
            "type": "Microsoft.MachineLearningServices/workspaces/connections",
            "properties": {
                "authType": "ProjectManagedIdentity",
                "category": "RemoteTool",
                "target": MCP_ENDPOINT,
                "isSharedToAll": True,
                "audience": "https://search.azure.com/",
                "metadata": {"ApiType": "Azure"},
            },
        }
        conn_url = f"https://management.azure.com{PROJECT_RESOURCE_ID}/connections/{PROJECT_CONNECTION_NAME}?api-version=2025-10-01-preview"
        
        response = requests.put(conn_url, headers=headers, json=body)
        response.raise_for_status()
        logger.info("Project Connection 설정 완료")

        # [B] 클라이언트 및 에이전트 설정
        project_client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT, 
            credential=credential
        )
        
        instructions = """
        너는 여행 전문 AI 에이전트야. 반드시 연결된 지식 기반 touragent 를 기반으로 응답해줘.
        혼자 생각해서 생성하지 말고, 항상 응답에 참조한 데이터를 언급해줘. 모르면 모른다고 답변해줘.
        """
        
        mcp_kb_tool = MCPTool(
            server_label="knowledge-base",
            server_url=MCP_ENDPOINT,
            require_approval="never",
            allowed_tools=["knowledge_base_retrieve"],
            project_connection_id=PROJECT_CONNECTION_NAME,
        )

        agent = project_client.agents.create_version(
            agent_name=AGENT_NAME,
            definition=PromptAgentDefinition(
                model=AGENT_MODEL,
                instructions=instructions,
                tools=[mcp_kb_tool],
            ),
        )
        
        logger.info(f"에이전트 생성 완료: {agent.name}")
        return project_client, agent
        
    except Exception as e:
        logger.error(f"에이전트 초기화 실패: {e}")
        raise e

# ============================================
# 5. 에이전트 로드
# ============================================
try:
    with st.spinner("🔄 AI 에이전트 초기화 중..."):
        project_client, agent = initialize_agent()
        openai_client = project_client.get_openai_client()
    st.success("✅ 에이전트 준비 완료!", icon="🤖")
except Exception as e:
    st.error(f"❌ 에이전트 초기화 실패: {e}")
    st.info("💡 Azure Portal에서 Web App의 Managed Identity가 활성화되어 있는지 확인하세요.")
    st.stop()

# ============================================
# 6. 세션 상태 관리
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    try:
        conv = openai_client.conversations.create()
        st.session_state.conversation_id = conv.id
    except Exception as e:
        st.error(f"대화 세션 생성 실패: {e}")
        st.stop()

# ============================================
# 7. UI: 기존 메시지 표시
# ============================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# 8. UI: 채팅 입력 및 스트리밍 응답
# ============================================
if prompt := st.chat_input("파리 여행에 대해 궁금한 점을 물어보세요!"):
    # 유저 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Responses API 스트리밍 호출
            stream = openai_client.responses.create(
                stream=True,
                conversation=st.session_state.conversation_id,
                tool_choice="required",
                input=prompt,
                extra_body={
                    "agent": {
                        "name": agent.name,
                        "type": "agent_reference",
                    }
                },
            )

            for event in stream:
                if event.type == "response.output_text.delta":
                    full_response += (event.delta or "")
                    response_placeholder.markdown(full_response + "▌")
                elif event.type == "response.completed":
                    response_placeholder.markdown(full_response)

            # 대화 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            st.error(f"응답 생성 중 오류가 발생했습니다: {e}")

# ============================================
# 9. 사이드바: 정보 및 세션 관리
# ============================================
with st.sidebar:
    st.header("ℹ️ 정보")
    st.write(f"**에이전트**: {AGENT_NAME}")
    st.write(f"**모델**: {AGENT_MODEL}")
    
    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        conv = openai_client.conversations.create()
        st.session_state.conversation_id = conv.id
        st.rerun()
