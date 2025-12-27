import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. 核心算法定义
# ==========================================

def predict_logit_model_simple(df, target_drug, new_share_target):
    # 过滤掉数据不完整的行，防止报错
    df = df.dropna(subset=['当前份额', '效用值']).copy()
    
    target_row = df[df['品牌'] == target_drug]
    if target_row.empty: return pd.Series(dtype=float)
    
    remaining_share = 1.0 - new_share_target
    other_drugs = df[df['品牌'] != target_drug].copy()
    
    # 核心逻辑：权重 = 当前份额 * 效用值
    other_drugs['weight'] = other_drugs['当前份额'] * other_drugs['效用值']
    total_weight = other_drugs['weight'].sum()
    
    if total_weight == 0:
        if other_drugs['当前份额'].sum() > 0:
            other_drugs['predicted_share'] = remaining_share * (other_drugs['当前份额'] / other_drugs['当前份额'].sum())
        else:
            other_drugs['predicted_share'] = 0
    else:
        other_drugs['predicted_share'] = remaining_share * (other_drugs['weight'] / total_weight)
        
    result = other_drugs.set_index('品牌')['predicted_share']
    result[target_drug] = new_share_target
    return result

def predict_proportional_model_simple(df, target_drug, new_share_target):
    df = df.dropna(subset=['当前份额']).copy()
    
    target_row = df[df['品牌'] == target_drug]
    if target_row.empty: return pd.Series(dtype=float)

    s_x_old = target_row['当前份额'].values[0]
    delta_s_x = new_share_target - s_x_old
    
    other_drugs = df[df['品牌'] != target_drug].copy()
    sum_s_i_old = other_drugs['当前份额'].sum()
    
    if sum_s_i_old == 0:
        other_drugs['predicted_share'] = 0
    else:
        other_drugs['predicted_share'] = other_drugs['当前份额'] - delta_s_x * (other_drugs['当前份额'] / sum_s_i_old)
    
    other_drugs['predicted_share'] = other_drugs['predicted_share'].apply(lambda x: max(0, x))
    
    result = other_drugs.set_index('品牌')['predicted_share']
    result[target_drug] = new_share_target
    return result

# ==========================================
# 2. 页面布局与交互设计
# ==========================================

st.set_page_config(page_title="市场份额模拟预测器", layout="wide", page_icon="📈")

st.title("📈 医药市场份额预测模拟器")
st.markdown("本应用基于 **Logit 效用模型** 与 **Proportional 比例模型**，帮助你模拟当某款产品市场份额发生变化时，对竞争格局的影响。")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    st.info("💡 **效用值 (Utility)** 代表产品的综合竞争力。值越高，在竞争中越容易维持份额。")

# --- Step 1: 数据录入 ---
st.subheader("1. 初始化市场数据")

if 'init_df' not in st.session_state:
    data = {
        '品牌': ['自家产品A', '竞品B', '竞品C', '竞品D'],
        '当前份额': [0.10, 0.40, 0.30, 0.20],
        '效用值': [1.0, 1.2, 0.9, 0.8], 
        '年治疗费用(元)': [5000, 6000, 4500, 5500]
    }
    st.session_state.init_df = pd.DataFrame(data)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("请在下方表格直接修改数据。**请确保‘当前份额’总和为 1.0**")
    edited_df = st.data_editor(
        st.session_state.init_df,
        num_rows="dynamic",
        column_config={
            "当前份额": st.column_config.NumberColumn(format="%.2f", min_value=0, max_value=1),
            "效用值": st.column_config.NumberColumn(help="Exp_Utility，必须大于0", min_value=0.01),
            "年治疗费用(元)": st.column_config.NumberColumn(format="￥%d")
        },
        use_container_width=True
    )

# 数据校验与清洗：将可能的 NaN 填充为 0
edited_df = edited_df.fillna(0) 
total_share = edited_df['当前份额'].sum()

with col2:
    st.metric("当前市场总份额", f"{total_share:.1%}")
    if not (0.99 <= total_share <= 1.01):
        st.error("⚠️ 错误：份额总和必须等于 100%")
        st.stop()
    else:
        st.success("✅ 数据校验通过")

# --- Step 2: 设定预测目标 ---
st.divider()
st.subheader("2. 设定预测目标")

# 确保有品牌数据，否则停止
if edited_df.empty or '品牌' not in edited_df.columns:
    st.warning("请先在表格中输入品牌数据")
    st.stop()

brand_options = edited_df['品牌'].astype(str).unique()
c1, c2, c3 = st.columns(3)

with c1:
    target_product = st.selectbox("选择自家产品", options=brand_options)

# 安全获取当前份额
current_rows = edited_df[edited_df['品牌']==target_product]
if not current_rows.empty:
    current_target_share = current_rows['当前份额'].values[0]
else:
    current_target_share = 0.0

with c2:
    new_share_input = st.slider(
        f"设定 '{target_product}' 的预期新份额",
        min_value=0.0, max_value=1.0, 
        value=min(1.0, float(current_target_share) + 0.05), 
        step=0.01,
        format="%.2f"
    )

with c3:
    algorithm = st.radio("选择预测算法", ["Logit模型 (基于效用)", "Proportional模型 (基于比例)"])

# --- Step 3: 计算与展示 ---
st.divider()

if st.button("🚀 开始预测", type="primary"):
    
    # 1. 运行计算
    if algorithm.startswith("Logit"):
        result_series = predict_logit_model_simple(edited_df, target_product, new_share_input)
    else:
        result_series = predict_proportional_model_simple(edited_df, target_product, new_share_input)
    
    # 2. 整理结果数据
    result_df = edited_df.copy()
    
    # 映射结果，如果没有预测值则填0
    result_df['预测新份额'] = result_df['品牌'].map(result_series).fillna(0)
    result_df['份额变化'] = result_df['预测新份额'] - result_df['当前份额']
    
    # BIA 计算
    old_avg_cost = (result_df['当前份额'] * result_df['年治疗费用(元)']).sum()
    new_avg_cost = (result_df['预测新份额'] * result_df['年治疗费用(元)']).sum()
    cost_change = new_avg_cost - old_avg_cost

    # 3. 核心指标展示
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(f"{target_product} 新份额", f"{new_share_input:.1%}", f"{(new_share_input - current_target_share):.1%}")
    kpi2.metric("人均治疗费用变化", f"￥{new_avg_cost:,.0f}", f"{cost_change:+,.0f}")
    
    # 找出受冲击最大的竞品
    competitors = result_df[result_df['品牌']!=target_product]
    if not competitors.empty:
        loser = competitors.sort_values('份额变化').iloc[0]
        kpi3.metric("受冲击最大竞品", f"{loser['品牌']}", f"{loser['份额变化']:.1%}")
    else:
        kpi3.metric("受冲击最大竞品", "无", "0%")

    # 4. 图表展示
    st.subheader("📊 预测结果可视化")
    tab1, tab2 = st.tabs(["市场格局对比", "份额变化瀑布图"])
    
    with tab1:
        plot_df = result_df[['品牌', '当前份额', '预测新份额']].melt(id_vars='品牌', var_name='状态', value_name='份额')
        fig = px.bar(plot_df, x='品牌', y='份额', color='状态', barmode='group', 
                     text_auto='.1%', 
                     color_discrete_map={'当前份额': '#d3d3d3', '预测新份额': '#1f77b4'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig_waterfall = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative"] * len(result_df),
            x = result_df['品牌'],
            textposition = "outside",
            text = [f"{x:.1%}" for x in result_df['份额变化']],
            y = result_df['份额变化'],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title = "份额流动分析")
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # 5. 详细数据表 (已修复Bug)
    st.subheader("📋 详细数据表")
    
    # 关键修复：再次填充空值，确保 style.format 不会因为 None 报错
    result_df_filled = result_df.fillna(0)
    
    st.dataframe(result_df_filled.style.format({
        '当前份额': '{:.2%}', 
        '预测新份额': '{:.2%}', 
        '份额变化': '{:+.2%}',
        '年治疗费用(元)': '￥{:,.0f}'
    }).background_gradient(subset=['份额变化'], cmap='RdYlGn'))