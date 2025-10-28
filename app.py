import streamlit as st
import pandas as pd
import os
import time

# 导入我们所有的自定义模块
try:
    from radar_sei_system.data_management import load_iq_data
    from radar_sei_system.feature_extraction import extract_features
    from radar_sei_system.ml_modeling import train, predict
    from radar_sei_system.performance_evaluation import evaluate
except ImportError as e:
    st.error(f"启动失败：无法导入核心模块。请检查 __init__.py 文件是否配置正确。")
    st.error(f"详细错误: {e}")
    st.stop()


# --- 1. 基本配置和常量 ---
st.set_page_config(page_title="雷达辐射源识别系统 (MVP)", layout="wide")
st.title("⚡ 雷达辐射源指纹识别系统 (MVP版)")

# 硬编码模型路径 (与ml_modeling模块中的DEFAULT_MODEL_DIR保持一致)
MODEL_SAVE_DIR = "./saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mvp_model.pkl")

# 临时的文件处理目录
TEMP_DIR = "./temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- 2. 辅助函数 ---
def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录，返回文件路径"""
    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def cleanup_temp_file(file_path):
    """删除临时文件"""
    if os.path.exists(file_path):
        os.remove(file_path)

# --- 3. 页面导航 (侧边栏) ---
st.sidebar.title("导航")
page = st.sidebar.radio("选择功能", ["🎯 预测 (Prediction)", "🏋️ 训练 (Training)"])

# ==============================================================================
# --- 页面一：预测 ---
# ==============================================================================
if page == "🎯 预测 (Prediction)":
    st.header("🎯 辐射源身份预测")

    # 1. 检查模型是否存在
    if not os.path.exists(MODEL_SAVE_PATH):
        st.error(f"警告：未找到已训练的模型！ (应位于: {MODEL_SAVE_PATH})")
        st.info("请先去 '🏋️ 训练 (Training)' 页面训练一个模型。")
        st.stop()
    
    st.success(f"已加载模型: {MODEL_SAVE_PATH}")

    # 2. 文件上传 (只接受 .h5)
    uploaded_file = st.file_uploader("上传一个 .h5 文件进行预测", type=["h5"])

    if uploaded_file is not None:
        st.write(f"已上传文件: `{uploaded_file.name}`")
        
        # 3. 开始预测
        if st.button("开始预测"):
            with st.spinner('正在处理...'):
                temp_file_path = ""
                try:
                    # 步骤 A: 保存临时文件
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    # 步骤 B: 加载数据
                    st.subheader("1. 数据加载")
                    data_obj = load_iq_data(temp_file_path)
                    if data_obj:
                        st.write(f"信号长度: {len(data_obj['iq_data'])}, 采样率: {data_obj['sampling_rate']/1e6} MHz")
                        st.write(f"文件内部标签 (仅供参考): {data_obj.get('label', 'N/A')}")
                    else:
                        st.error("数据加载失败！请检查文件格式是否正确。")
                        st.stop()

                    # 步骤 C: 提取特征
                    st.subheader("2. 特征提取")
                    # MVP阶段，我们只使用 'power_spectrum' 特征
                    feature_obj = extract_features(data_obj, methods=['power_spectrum'])
                    if not feature_obj.empty:
                        st.dataframe(feature_obj)
                    else:
                        st.error("特征提取失败！")
                        st.stop()

                    # 步骤 D: 执行预测
                    st.subheader("3. 预测结果")
                    prediction_list = predict(feature_obj, MODEL_SAVE_PATH)
                    if prediction_list:
                        result = prediction_list[0]
                        st.metric(label="预测标签", value=result.get('predicted_label'))
                        st.json(result.get('probabilities'))
                    else:
                        st.error("预测执行失败！")
                        st.stop()

                except Exception as e:
                    st.error(f"处理过程中发生严重错误: {e}")
                
                finally:
                    # 步骤 E: 清理临时文件
                    cleanup_temp_file(temp_file_path)
                    
            st.success("预测完成！")

# ==============================================================================
# --- 页面二：训练 ---
# ==============================================================================
elif page == "🏋️ 训练 (Training)":
    st.header("🏋️ 训练新模型")
    st.info("""
    **重要提示 (MVP版):**
    1.  请上传**多个** .h5 训练文件。
    2.  为确保分类器能工作，上传的文件必须包含**至少2个不同**的内部标签 (例如: 一些是'1021', 一些是'1022')。
    3.  程序将自动从 .h5 文件内部的 `InterPulse/LABEL` 路径读取标签。
    """)

    # 1. 文件上传 (多文件, 只接受 .h5)
    uploaded_files = st.file_uploader("上传训练数据集 (可多选)", accept_multiple_files=True, type=["h5"])

    if uploaded_files:
        st.write(f"总共上传了 {len(uploaded_files)} 个文件。")
        
        # 2. 开始训练
        if st.button("开始训练"):
            if len(uploaded_files) < 2:
                st.error("训练至少需要2个文件。")
                st.stop()

            all_features_list = []
            all_labels = []
            temp_files_to_clean = []
            
            with st.spinner(f'正在处理 {len(uploaded_files)} 个文件... 这可能需要几分钟...'):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"正在处理: {file.name}...")
                    try:
                        # =================================================
                        # --- 这是我们修正的关键逻辑 ---
                        # =================================================
                        
                        # 步骤 A: 保存临时文件
                        temp_path = save_uploaded_file(file)
                        temp_files_to_clean.append(temp_path)
                        
                        # 步骤 B: 加载数据 (loader.py 会自动提取内部标签)
                        data_obj = load_iq_data(temp_path)
                        if not data_obj:
                            st.warning(f"加载 {file.name} 失败，已跳过。")
                            continue
                            
                        # 步骤 C: 提取标签 (从 data_obj 中获取)
                        label = data_obj.get("label")
                        if not label or label == "unknown":
                            st.warning(f"文件 {file.name} 内部未找到有效标签，已跳过。")
                            continue
                            
                        # 步骤 D: 提取特征
                        feature_obj = extract_features(data_obj, methods=['power_spectrum'])
                        if feature_obj.empty:
                            st.warning(f"提取 {file.name} 特征失败，已跳过。")
                            continue
                        
                        # =================================================
                        # --- 修正逻辑结束 ---
                        # =================================================
                            
                        all_features_list.append(feature_obj)
                        all_labels.append(label)
                        
                    except Exception as e:
                        st.warning(f"处理 {file.name} 时出错: {e}，已跳过。")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("所有文件处理完毕！")

            # 步骤 E: 合并所有特征
            if not all_features_list:
                st.error("没有文件被成功处理！请检查文件格式和内容。")
                st.stop()
                
            training_features = pd.concat(all_features_list, ignore_index=True)
            st.subheader("提取的特征总览 (前5行):")
            st.dataframe(training_features.head())
            
            label_counts = pd.Series(all_labels).value_counts()
            st.write(f"总共提取了 {len(all_labels)} 个样本。")
            st.write("标签分布:")
            st.dataframe(label_counts)

            if len(label_counts) < 2:
                st.error(f"训练失败：只找到了 {len(label_counts)} 个唯一的标签。分类器至少需要2个不同的类别才能训练。")
                st.stop()

            # 步骤 F: 执行训练
            st.subheader("模型训练")
            with st.spinner("正在训练模型..."):
                model_path, train_log = train(training_features, all_labels, 
                                              model_type="mvp_logistic", params={})
            
            st.success(f"训练完成！模型已保存到: {model_path}")
            st.json(train_log)

            # 步骤 G: (可选) 在训练集上进行评估
            st.subheader("训练集表现 (用于调试)")
            predictions_on_train = predict(training_features, model_path)
            eval_results = evaluate(predictions_on_train, all_labels)
            
            st.metric(label="训练集准确率", value=f"{eval_results.get('accuracy', 0):.2%}")
            
            st.write("混淆矩阵:")
            cm_labels = eval_results.get('labels_in_matrix')
            if cm_labels:
                st.dataframe(pd.DataFrame(eval_results.get('confusion_matrix'), 
                                         columns=cm_labels, 
                                         index=cm_labels))
            else:
                st.write("无法生成混淆矩阵。")

            # 步骤 H: 清理所有临时文件
            for path in temp_files_to_clean:
                cleanup_temp_file(path)
            st.info("临时文件已清理。")