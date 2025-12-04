import streamlit as st
import tempfile
import time
from pathlib import Path
from PIL import Image

# 导入我们的包
from packages.detector import YOLOv5Detector
from packages.pdf_processor import PDFProcessor
from packages.ocr_engine import OCREngine
from packages.visualizer import Visualizer
from packages.models import PageResult

# 设置页面配置
st.set_page_config(page_title="YOLOv5 PDF Detector", layout="wide")

# 初始化 Session State
if 'detector' not in st.session_state:
    # 权重文件应当在当前目录的 weights 文件夹下
    st.session_state.detector = YOLOv5Detector(weights_path='weights/best.pt')

def process_file(uploaded_file):
    """主处理流程，对应 Detector.swift 的 processPDF"""
    
    st.info("正在读取 PDF...")
    bytes_data = uploaded_file.getvalue()
    
    # 1. PDF 转图片
    # 注意：Swift 中采用了双管线 (DetectImage vs HDImage)，这里为简化直接用高清图 resize
    hd_images = PDFProcessor.pdf_to_images(bytes_data, dpi=200) # dpi 200 约等于高清
    
    if not hd_images:
        st.error("无法解析 PDF")
        return []

    page_results = []
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_pages = len(hd_images)
    
    for i, hd_img in enumerate(hd_images):
        status_text.text(f"处理第 {i+1}/{total_pages} 页...")
        
        # 2. 推理
        # Swift 中的 detectImage 只有 640x640 左右，这里我们不用显式缩放，
        # YOLOv5 的 forward 会自动 resize，但我们可以手动 resize 以加速传输
        detect_img = hd_img.copy() # 如果太大可以 resize
        
        detections = st.session_state.detector.detect(detect_img)
        
        # 3. 绘图
        annotated_img = Visualizer.draw_annotations(hd_img, detections)
        
        # 4. 裁剪
        crops = Visualizer.crop_detections(hd_img, detections)
        
        # 5. OCR
        captions = OCREngine.recognize_batch(crops)
        
        # 保存结果
        res = PageResult(
            page_index=i,
            annotated_image=annotated_img,
            crops=crops,
            captions=captions
        )
        page_results.append(res)
        
        progress_bar.progress((i + 1) / total_pages)

    status_text.text("处理完成")
    return page_results

def main():
    st.title("📄 YOLOv5 PDF Figure Detector")
    
    # 侧边栏上传
    with st.sidebar:
        st.header("上传文件")
        uploaded_file = st.file_uploader("选择 PDF 文件", type=['pdf'])
        
        if uploaded_file is not None:
            if st.button("开始检测"):
                with st.spinner("AI 正在思考..."):
                    results = process_file(uploaded_file)
                    st.session_state.results = results

    # 显示结果区域 (对应 ResultsView)
    if 'results' in st.session_state and st.session_state.results:
        st.divider()
        st.subheader("检测结果")
        
        # 选项卡显示每一页
        tabs = st.tabs([f"第 {r.page_index + 1} 页" for r in st.session_state.results])
        
        for tab, result in zip(tabs, st.session_state.results):
            with tab:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(result.annotated_image, caption="检测总览", width='stretch')
                
                with col2:
                    st.write("##### 提取的插图与 OCR")
                    if not result.crops:
                        st.info("本页未检测到目标")
                    else:
                        for crop, cap in zip(result.crops, result.captions):
                            with st.container(border=True):
                                st.image(crop, width='stretch')
                                if cap:
                                    st.caption(f"📝 {cap}")
                                else:
                                    st.caption("无文字内容")

if __name__ == "__main__":
    main()