"""
导入流程主图

使用 LangGraph 构建文档导入工作流
"""
import json

from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from processor.import_process.nodes.bge_embedding import BgeEmbeddingNode
from processor.import_process.nodes.document_split import DocumentSplitNode
from processor.import_process.nodes.entry import EntryNode
from processor.import_process.nodes.import_milvus import ImportMilvusNode
from processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from processor.import_process.nodes.md_img import MdImgNode
from processor.import_process.nodes.pdf_to_md import PdfToMdNode
from processor.import_process.state import ImportGraphState, create_default_state


# router
def router(state: ImportGraphState) -> str:
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return END


def import_router(state: ImportGraphState) -> str:
    """
    入口节点后的路由逻辑
    根据文件类型决定走 PDF 转换分支还是直接处理 MD 分支

    Args:
        state: 当前图状态

    Returns:
        下一个节点名称
    """
    if state.get('is_md_read_enabled'):
        return "md_img_node"
    if state.get('is_pdf_read_enabled'):
        return "pdf_to_md_node"
    return END  # 安全降级


def create_import_graph() -> CompiledStateGraph:
    # 1. 定义状态图
    graph = StateGraph(ImportGraphState)

    # 2. 定义节点
    nodes = {
        "bge_embedding_node": BgeEmbeddingNode(),
        "document_split_node": DocumentSplitNode(),
        "entry_node": EntryNode(),
        "import_milvus_node": ImportMilvusNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "md_img_node": MdImgNode(),
        "pdf_to_md_node": PdfToMdNode(),
    }

    graph.set_entry_point("entry_node")

    for key, value in nodes.items():
        graph.add_node(key, value)

    # 条件边，路由
    graph.add_conditional_edges("entry_node", router, {
        "md_img_node": "md_img_node",
        "pdf_to_md_node": "pdf_to_md_node",
        END: END
    })
    graph.add_edge("pdf_to_md_node", "md_img_node")
    graph.add_edge("md_img_node", "document_split_node")
    graph.add_edge("document_split_node", "item_name_recognition_node")
    graph.add_edge("item_name_recognition_node", "bge_embedding_node")
    graph.add_edge("bge_embedding_node", "import_milvus_node")

    # 4. 编译图
    return graph.compile()


# 创建全局图实例
kb_import_graph = graph_compiled = create_import_graph()

if __name__ == '__main__':
    # setup_logging()
    state = create_default_state(
        # is_pdf_read_enabled=True,
        # is_md_read_enabled=False,
        import_file_plugins="E:/gsq/sgg/classstudy/11_尚硅谷AI全能开发技术之项目【掌柜智库】/2.资料/pdf文档/doc/万用表RS-12的使用.pdf",
        file_dir="E:/temp_dir"
    )

    init_state = create_default_state(**state)

    final_state = None

    for event in kb_import_graph.stream(state):
        for node_name, state in event.items():
            print(f"节点的名称是：{node_name}")
            final_state = state

    print(f"最终状态：{final_state}")
    print(json.dumps(final_state, indent=4, ensure_ascii=False))

    kb_import_graph.get_graph().print_ascii()
