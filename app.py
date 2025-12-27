import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. 核心算法定义
# ==========================================

def predict_proportional_model_simple(df, target_drug, new_share_target):
    """传统等比例损失法"""
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

def predict_logit_model_simple(df, target_drug, new_share_target):
    """传统 Logit 法"""
    df = df.dropna(subset=['当前份额', '指数效用值']).copy()
    remaining_share = 1.0 - new_share_target
    other_drugs = df[df['品牌'] != target_drug].copy()
    
    other_drugs['weight'] = other_drugs['当前份额'] * other_drugs['指数效用值']
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

def predict_bayesian_replacement(df, target_drug, new_share_target):
    """【论文标准版】贝叶斯算法: Likelihood = Ui / (Ui + Ux)"""
    return predict_bayesian_custom(df, target_drug, new_share_target, formula_str="Ui / (Ui + Ux)")

def predict_bayesian_custom(df, target_drug, new_share_target, formula_str):
    """
    【自定义版】贝叶斯算法
    允许用户输入 Likelihood 的计算公式。
    """
    df = df.dropna(subset=['当前份额', '指数效用值']).copy()
    
    target_row = df[df['品牌'] == target_drug]
    if target_row.empty: return pd.Series(dtype=float), None
    
    Ux = target_row['指数效用值'].values[0]
    Px = target_row['年治疗费用(元)'].values[0]
    
    other_drugs = df[df['品牌'] != target_drug].copy()
    
    likelihoods = []
    try:
        for index, row in other_drugs.iterrows():
            Ui = row['指数效用值']
            Pi = row['年治疗费用(元)']
            Si = row['当前份额']
            
            # 安全执行公式
            val = eval(formula_str, {"__builtins__": None}, {
                "Ui": Ui, "Ux": Ux, 
                "Pi": Pi, "Px": Px, 
                "Si": Si, "np": np
            })
            likelihoods.append(max(0.0, float(val)))
            
        other_drugs['Likelihood'] = likelihoods
        
    except Exception as e:
        st.error(f"公式解析失败: {e}")
        return pd.Series(dtype=float), None

    other_drugs['Posterior_Numerator'] = other_drugs['当前份额'] * other_drugs['Likelihood']
    denominator = other_drugs['Posterior_Numerator'].sum()
    remaining_share = 1.0 - new_share_target
    
    if denominator == 0:
        sum_orig = other_drugs['当前份额'].sum()
        if sum_orig > 0:
            other_drugs['predicted_share'] = remaining_share * (other_drugs['当前份额'] / sum_orig)
        else:
            other_drugs['predicted_share'] = 0
    else:
        other_drugs['predicted_share'] = remaining_share * (other_drugs['Posterior_Numerator'] / denominator)
    
    result = other_drugs.set_index('品牌')['predicted_share']
    result[target_drug] = new_share_target
    
    return result, other_drugs[['品牌', 'Likelihood']]

# ==========================================
# 2. 页面布局与交互设计
# ==========================================

st.set_page_config(page_title="基于DCE指数效用值市场份额预测器", layout="wide", page_icon="🧬")

st.title("🧬 基于DCE指数效用值市场份额预测器 ")
st.markdown("""
本工具基于 **贝叶斯推断框架**，通过定义 **似然函数 (Likelihood)** 来模拟竞品在面对新产品冲击时的保留概率。
支持使用预设标准算法，或自定义似然逻辑。
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 变量词典")
    st.markdown("""
    在自定义公式中，你可以使用以下变量：
    - **`Ui`**: 竞品 $i$ 的指数效用值
    - **`Ux`**: 目标产品 $x$ 的指数效用值
    - **`Pi`**: 竞品 $i$ 的年费用
    - **`Px`**: 目标产品 $x$ 的年费用
    - **`Si`**: 竞品 $i$ 的当前份额
    - **`np`**: Numpy数学库
    """)

# --- Step 1: 数据录入 ---
st.subheader("1. 市场数据初始化")

# 定义标准列名
REQUIRED_COLUMNS = ['品牌', '当前份额', '指数效用值', '年治疗费用(元)']

# 1. 检查 Session State 中的数据是否合法 (是否缺失关键列)
# 如果是旧缓存导致的数据不一致，强制重置
need_reset = False
if 'init_df' not in st.session_state:
    need_reset = True
else:
    # 检查当前内存中的 df 是否包含所有必须的列
    existing_cols = st.session_state.init_df.columns.tolist()
    if not all(col in existing_cols for col in REQUIRED_COLUMNS):
        need_reset = True

# 2. 如果需要，初始化/重置数据
if need_reset:
    data = {
        '品牌': ['自家产品', '竞品A', '竞品B', '竞品C'],
        '当前份额': [0.54, 0.22, 0.19, 0.05], 
        '指数效用值': [2.58, 0.88, 1.41, 0.50], 
        '年治疗费用(元)': [6000, 8000, 7500, 5000]
    }
    st.session_state.init_df = pd.DataFrame(data)
    # 强制刷新页面以应用重置
    st.rerun()

col1, col2 = st.columns([2, 1])
with col1:
    edited_df = st.data_editor(
        st.session_state.init_df,
        num_rows="dynamic", # 允许添加新行
        column_config={
            "当前份额": st.column_config.NumberColumn(format="%.2f", min_value=0, max_value=1),
            "指数效用值": st.column_config.NumberColumn(label="指数效用值(U)", min_value=0.01),
            "年治疗费用(元)": st.column_config.NumberColumn(format="￥%d")
        },
        use_container_width=True
    ).fillna(0) # 填充新加行的空值为0

    # 【关键修复】过滤掉新加行中“品牌”为空的数据，防止后续报错
    # 只有当品牌名不为空，且不为0时，才视为有效数据
    edited_df = edited_df[edited_df['品牌'].astype(str).str.strip() != '0']
    edited_df = edited_df[edited_df['品牌'].astype(str).str.strip() != '']
    edited_df = edited_df[edited_df['品牌'].notna()]

total_share = edited_df['当前份额'].sum()
with col2:
    st.metric("当前市场总份额", f"{total_share:.1%}")
    if not (0.99 <= total_share <= 1.01):
        st.error("⚠️ 份额总和必须等于 100%")
        st.stop()
    else:
        st.success("✅ 校验通过")

   
# --- Step 2: 设定目标 ---
st.divider()
st.subheader("2. 设定预测情景")

if edited_df.empty: st.stop()

c1, c2 = st.columns(2)
with c1:
    target_product = st.selectbox("选择目标产品 (x)", options=edited_df['品牌'])
    tgt_row = edited_df[edited_df['品牌']==target_product]
    curr_share = tgt_row['当前份额'].values[0] if not tgt_row.empty else 0
    Ux_val = tgt_row['指数效用值'].values[0] if not tgt_row.empty else 1.0

with c2:
    new_share_input = st.slider(
        f"设定 {target_product} 的新市场份额",
        0.0, 1.0, float(min(1.0, curr_share + 0.1)), 0.01
    )

st.write("---")
st.subheader("3. 选择预测算法")

algo_type = st.radio(
    "选择似然函数 (Likelihood Function) 的定义方式：",
    ["基于DCE指数效用值模式 (Ui / (Ui + Ux))", 
     "自定义模式 (编写Python表达式)", 
     "传统等比例损失", 
     "传统多项 Logit"],
    index=0
)

# --- 自定义公式区域 ---
custom_formula = "Ui / (Ui + Ux)" # 默认值

if algo_type == "自定义模式 (编写Python表达式)":
    # 创建两列，左边输入，右边放一点提示
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        custom_formula = st.text_input(
            "输入 Python 表达式 (返回似然概率)", 
            value="Ui / (Ui + Ux)"
        )
    
    with col_f2:
        st.info("💡 记得使用 sidebar 中的变量名 (Ui, Ux, Pi...)")

    # --- 这里是你要的新增部分：使用折叠面板提供详细指南 ---
    with st.expander("📖 如何使用“自定义似然函数”？(点击展开高级指南)"):
        st.markdown("""
        ### 🧠 什么是“似然函数” (Likelihood)?
        在贝叶斯框架下，`Likelihood` 代表 **“在面对目标产品 x 的冲击时，竞品 i 保留住份额的概率”**。
        - 结果接近 **1.0**: 竞品防御力强，几乎不流失份额。
        - 结果接近 **0.0**: 竞品防御力弱，份额大量流失给新产品。

        ### 📝 常用公式示例 (可直接复制到输入框)

        **1. 论文标准模型 (基于效用)**
        - **公式**: `Ui / (Ui + Ux)`
        - **含义**: 基于DCE指数效用值模型。仅看效用对比，效用越高，保留概率越大。

        **2. 性价比模型 (Cost-Effectiveness)**
        - **公式**: `(Ui/Pi) / ((Ui/Pi) + (Ux/Px))` 或 `(Pi/Ui) / ((Pi/Ui) + (Px/Ux))`
        - **含义**: 假设医生决策是基于“每元钱买到的疗效”或“单位效用值成本”。如果竞品性价比(U/P)更高或单位效用成本(P/U)更低，则更容易保留。

        **3. 价格敏感模型 (纯价格防御)**
        - **公式**: `(1/Pi) / ((1/Pi) + (1/Px))`
        - **含义**: 假设市场对价格极度敏感。价格越低(1/P越大)，保留概率越高。

        **4. 赢家通吃模型 (Winner Takes All)**
        - **公式**: `1.0 if Ui >= Ux else 0.0`
        - **含义**: 激进假设。只要竞品效用比新药高，一点份额都不丢；只要比新药低，在该轮竞争中全部流失。
        """)

# --- Step 3: 计算与展示 ---
st.divider()

if st.button("🚀 开始预测", type="primary"):
    
    likelihood_data = None
    
    if "论文标准" in algo_type:
        result_series, likelihood_data = predict_bayesian_replacement(edited_df, target_product, new_share_input)
    elif "自定义" in algo_type:
        result_series, likelihood_data = predict_bayesian_custom(edited_df, target_product, new_share_input, custom_formula)
    elif "等比例" in algo_type:
        result_series = predict_proportional_model_simple(edited_df, target_product, new_share_input)
    else:
        result_series = predict_logit_model_simple(edited_df, target_product, new_share_input)
        
    if result_series.empty:
        st.stop()

    # 结果整合
    result_df = edited_df.copy()
    result_df['预测新份额'] = result_df['品牌'].map(result_series).fillna(0)
    result_df['份额变化'] = result_df['预测新份额'] - result_df['当前份额']
    
    if likelihood_data is not None:
        result_df = result_df.merge(likelihood_data, on='品牌', how='left')
        result_df.loc[result_df['品牌']==target_product, 'Likelihood'] = np.nan

    # KPI
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(f"{target_product} 新份额", f"{new_share_input:.1%}", f"{(new_share_input - curr_share):.1%}")
    cost_diff = (result_df['预测新份额']*result_df['年治疗费用(元)']).sum() - \
                (result_df['当前份额']*result_df['年治疗费用(元)']).sum()
    kpi2.metric("人均治疗费用变化", f"￥{cost_diff:+,.0f}")
    
    competitors = result_df[result_df['品牌']!=target_product]
    if not competitors.empty:
        loser = competitors.sort_values('份额变化').iloc[0]
        kpi3.metric("受冲击最大竞品", f"{loser['品牌']}", f"{loser['份额变化']:.1%}")

    # 可视化
    tab1, tab2 = st.tabs(["📊 结果图表", "📋 详细数据与似然验证"])
    
    with tab1:
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            plot_df = result_df[['品牌', '当前份额', '预测新份额']].melt(id_vars='品牌')
            fig = px.bar(plot_df, x='品牌', y='value', color='variable', barmode='group', text_auto='.1%')
            st.plotly_chart(fig, use_container_width=True)
        with c_chart2:
            fig_w = go.Figure(go.Waterfall(
                x=result_df['品牌'], y=result_df['份额变化'],
                text=[f"{v:+.1%}" for v in result_df['份额变化']],
                measure=["relative"]*len(result_df)
            ))
            fig_w.update_layout(title="份额净变化 (Waterfall)")
            st.plotly_chart(fig_w, use_container_width=True)

    with tab2:
        st.markdown(f"**当前使用的算法/公式:** `{algo_type if '自定义' not in algo_type else custom_formula}`")
        if likelihood_data is not None:
            st.caption("Likelihood 越大，说明竞品防御力越强，越难被目标产品取代。")
            
        fmt_dict = {
            '当前份额': '{:.2%}', '预测新份额': '{:.2%}', '份额变化': '{:+.2%}',
            '指数效用值': '{:.2f}', '年治疗费用(元)': '￥{:,.0f}'
        }
        if "Likelihood" in result_df.columns:
            fmt_dict['Likelihood'] = '{:.2%}'
            
        st.dataframe(result_df.style.format(fmt_dict).background_gradient(subset=['份额变化'], cmap='RdYlGn'))
