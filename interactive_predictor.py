"""
Interactive HEO Crystal Structure Predictor
============================================
Choose cation concentrations and predict crystal structure formation
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from feature_engineering import calculate_all_features, features_to_array
from elemental_properties import get_available_elements, ELEMENTAL_PROPERTIES
from optimize_composition import optimize_composition, multi_structure_optimization

# Page config
st.set_page_config(
    page_title="HEO Structure Predictor",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid;
        margin: 10px 0;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .fluorite {
        background-color: #E3F2FD;
        border-color: #2196F3;
        color: #1565C0;
    }
    .rock-salt {
        background-color: #FFF3E0;
        border-color: #FF9800;
        color: #E65100;
    }
    .spinel {
        background-color: #E8F5E9;
        border-color: #4CAF50;
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('heo_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please train the model first using train_model.py")
        return None

# Structure info
STRUCTURES = {
    0: {
        'name': 'Fluorite (α-PbO₂)',
        'short': 'Fluorite',
        'color': '#2196F3',
        'css_class': 'fluorite',
        'description': 'CaF₂-type structure with cubic symmetry',
        'characteristics': [
            'r_A/r_C < 0.35 (typically)',
            'Low electronegativity difference',
            'High coordination number',
            'Common in rare-earth oxides'
        ]
    },
    1: {
        'name': 'Rock-salt (Baddeleyite)',
        'short': 'Rock-salt',
        'color': '#FF9800',
        'css_class': 'rock-salt',
        'description': 'NaCl-type structure with cubic symmetry',
        'characteristics': [
            'r_A/r_C ~ 0.41 (typically)',
            'Moderate electronegativity difference',
            '6-fold coordination',
            'Common in transition metal oxides'
        ]
    },
    2: {
        'name': 'Spinel (Rutile)',
        'short': 'Spinel',
        'color': '#4CAF50',
        'css_class': 'spinel',
        'description': 'TiO₂-type structure',
        'characteristics': [
            'r_A/r_C ~ 0.45 (typically)',
            'Higher electronegativity difference',
            'Mixed coordination',
            'Common in complex oxides'
        ]
    }
}

def main():
    # Header
    st.markdown('<div class="main-header">🔬 HEO Crystal Structure Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict crystal structures based on cation concentrations</div>', unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        return

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Find Optimal Composition", "Quick Predict", "Concentration Sweep", "Batch Screening"]
    )

    # Available elements
    available_elements = get_available_elements()

    if mode == "Find Optimal Composition":
        optimal_composition_mode(model, available_elements)
    elif mode == "Quick Predict":
        quick_predict_mode(model, available_elements)
    elif mode == "Concentration Sweep":
        concentration_sweep_mode(model, available_elements)
    else:
        batch_screening_mode(model, available_elements)


def optimal_composition_mode(model, available_elements):
    """Find optimal proportions given elements"""
    st.header("🎯 Find Optimal Composition")
    st.info("Select elements and the system will find the optimal proportions for maximum stability")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Select Elements")

        # Number of elements
        n_elements = st.slider("Number of elements:", 2, 5, 4, key="opt_n")

        # Element selection
        selected_elements = []
        for i in range(n_elements):
            elem = st.selectbox(
                f"Element {i+1}:",
                available_elements,
                key=f"opt_elem_{i}",
                index=min(i, len(available_elements)-1)
            )
            selected_elements.append(elem)

        # Check for duplicates
        if len(selected_elements) != len(set(selected_elements)):
            st.error("⚠️ Duplicate elements detected! Please select unique elements.")
            return

        # Optimization target
        st.subheader("Optimization Target")
        target_choice = st.radio(
            "Optimize for:",
            ["Any stable structure (highest confidence)", "Specific structure"],
            key="opt_target"
        )

        target_structure = None
        if target_choice == "Specific structure":
            struct_map = {"Fluorite": 0, "Rock-salt": 1, "Spinel": 2}
            target_name = st.selectbox("Target structure:", list(struct_map.keys()))
            target_structure = struct_map[target_name]

        # Optimization method
        method = st.selectbox(
            "Optimization method:",
            ["auto", "gradient", "evolutionary"],
            help="Auto: gradient for ≤4 elements, evolutionary otherwise"
        )

        # Optimize button
        optimize_btn = st.button("🔬 Find Optimal Composition", type="primary", use_container_width=True)

    with col2:
        if optimize_btn:
            with st.spinner("Optimizing composition... This may take a moment..."):
                try:
                    # Run optimization
                    result = optimize_composition(
                        model,
                        selected_elements,
                        target_structure=target_structure,
                        method=method
                    )

                    # Display results
                    st.subheader("✅ Optimal Composition Found!")

                    # Composition table
                    st.write("**Optimal Proportions:**")
                    comp_df = pd.DataFrame({
                        'Element': selected_elements,
                        'Fraction': [f"{f:.4f}" for f in result['optimal_fractions']],
                        'Percentage': [f"{f*100:.2f}%" for f in result['optimal_fractions']],
                        'Molar': [f"{f:.4f}" for f in result['optimal_fractions']]
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    # Formula representation
                    formula = " + ".join([
                        f"{frac:.3f}{elem}"
                        for elem, frac in zip(selected_elements, result['optimal_fractions'])
                    ])
                    st.info(f"**Formula:** {formula}")

                    # Predicted structure
                    st.subheader("🔮 Predicted Structure")
                    struct_info = STRUCTURES[result['predicted_structure']]
                    st.markdown(
                        f'<div class="prediction-box {struct_info["css_class"]}">'
                        f'{struct_info["name"]}</div>',
                        unsafe_allow_html=True
                    )

                    # Confidence
                    confidence = result['confidence'] * 100
                    if confidence > 80:
                        st.success(f"✅ Very High Confidence: {confidence:.1f}%")
                    elif confidence > 70:
                        st.success(f"✅ High Confidence: {confidence:.1f}%")
                    elif confidence > 50:
                        st.warning(f"⚠️ Medium Confidence: {confidence:.1f}%")
                    else:
                        st.error(f"❌ Low Confidence: {confidence:.1f}%")

                    # Structure probabilities
                    st.subheader("📊 All Structure Probabilities")
                    for i, (struct_name, prob) in enumerate(
                        zip(['Fluorite', 'Rock-salt', 'Spinel'], result['structure_probabilities'])
                    ):
                        st.progress(prob, text=f"{struct_name}: {prob*100:.1f}%")

                    # Features
                    with st.expander("🔍 Calculated Features"):
                        features = result['features']
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("r_A/r_C", f"{features['r_A_r_C']:.4f}")
                            st.metric("Δχ (Pauling)", f"{features['delta_chi_pauling']:.4f}")
                            st.metric("Δχ (Mulliken)", f"{features['delta_chi_mulliken']:.4f}")
                        with col_b:
                            st.metric("Δδ (size)", f"{features['delta_size']:.4f}")
                            st.metric("ΔS_mix (J/mol·K)", f"{features['entropy_mixing']:.2f}")
                            st.metric("Components", f"{features['n_components']}")

                        # Design rules check
                        st.write("**📐 Design Rules Check:**")
                        r_ratio = features['r_A_r_C']
                        if r_ratio < 0.35:
                            st.info("✓ r_A/r_C < 0.35 → Fluorite favorable")
                        elif r_ratio > 0.45:
                            st.info("✓ r_A/r_C > 0.45 → Spinel favorable")
                        else:
                            st.info("✓ r_A/r_C ~ 0.35-0.45 → Rock-salt favorable")

                    # Multi-structure optimization option
                    st.subheader("🔄 Find Compositions for All Structures")
                    if st.button("Find optimal for Fluorite, Rock-salt, AND Spinel"):
                        with st.spinner("Optimizing for all structures..."):
                            multi_results = multi_structure_optimization(model, selected_elements)

                            st.write("**Optimal compositions for each structure:**")
                            for struct_name, res in multi_results.items():
                                with st.expander(f"**{struct_name}** (Confidence: {res['confidence']*100:.1f}%)"):
                                    comp_str = " + ".join([
                                        f"{frac:.3f}{elem}"
                                        for elem, frac in zip(selected_elements, res['optimal_fractions'])
                                    ])
                                    st.write(f"Composition: {comp_str}")

                                    comp_df = pd.DataFrame({
                                        'Element': selected_elements,
                                        'Fraction': [f"{f:.4f}" for f in res['optimal_fractions']],
                                        'Percentage': [f"{f*100:.2f}%" for f in res['optimal_fractions']]
                                    })
                                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def quick_predict_mode(model, available_elements):
    """Simple prediction interface"""
    st.header("🎯 Quick Prediction")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Select Cations & Concentrations")

        # Number of components
        n_components = st.slider("Number of cations:", 2, 5, 4)

        # Element selection and concentration input
        elements = []
        fractions = []

        for i in range(n_components):
            col_a, col_b = st.columns([2, 1])

            with col_a:
                elem = st.selectbox(
                    f"Cation {i+1}:",
                    available_elements,
                    key=f"elem_{i}",
                    index=min(i, len(available_elements)-1)
                )
                elements.append(elem)

            with col_b:
                frac = st.number_input(
                    f"Fraction:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/n_components,
                    step=0.05,
                    key=f"frac_{i}",
                    format="%.3f"
                )
                fractions.append(frac)

        # Show total
        total_frac = sum(fractions)
        if abs(total_frac - 1.0) > 0.01:
            st.warning(f"⚠️ Total fraction: {total_frac:.3f} (should be 1.0)")
        else:
            st.success(f"✓ Total fraction: {total_frac:.3f}")

        # Predict button
        predict = st.button("🔮 Predict Structure", type="primary", use_container_width=True)

    with col2:
        if predict:
            # Check for duplicate elements
            if len(elements) != len(set(elements)):
                st.error("⚠️ Duplicate elements detected! Please select unique cations.")
                return

            # Normalize fractions
            total = sum(fractions)
            if total > 0:
                fractions = [f/total for f in fractions]

            # Calculate features
            try:
                features_dict = calculate_all_features(elements, fractions)
                features_array = features_to_array(features_dict)

                # Predict
                prediction = model.predict([features_array])[0]
                probabilities = model.predict_proba([features_array])[0]

                # Display composition
                st.subheader("📋 Composition")
                comp_df = pd.DataFrame({
                    'Cation': elements,
                    'Fraction': [f"{f:.3f}" for f in fractions],
                    'Percentage': [f"{f*100:.1f}%" for f in fractions]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Display prediction
                st.subheader("🎯 Predicted Structure")
                struct_info = STRUCTURES[prediction]
                st.markdown(
                    f'<div class="prediction-box {struct_info["css_class"]}">'
                    f'{struct_info["name"]}</div>',
                    unsafe_allow_html=True
                )

                # Confidence
                confidence = probabilities[prediction] * 100
                if confidence > 70:
                    st.success(f"✅ High Confidence: {confidence:.1f}%")
                elif confidence > 50:
                    st.warning(f"⚠️ Medium Confidence: {confidence:.1f}%")
                else:
                    st.error(f"❌ Low Confidence: {confidence:.1f}%")

                # Probabilities
                st.subheader("📊 Structure Probabilities")
                for i, (struct_name, prob) in enumerate(zip(['Fluorite', 'Rock-salt', 'Spinel'], probabilities)):
                    st.progress(prob, text=f"{struct_name}: {prob*100:.1f}%")

                # Display features
                with st.expander("🔍 Calculated Features"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("r_A/r_C", f"{features_dict['r_A_r_C']:.4f}")
                        st.metric("Δχ (Pauling)", f"{features_dict['delta_chi_pauling']:.4f}")
                        st.metric("Δχ (Mulliken)", f"{features_dict['delta_chi_mulliken']:.4f}")
                    with col_b:
                        st.metric("Δδ (size)", f"{features_dict['delta_size']:.4f}")
                        st.metric("ΔS_mix (J/mol·K)", f"{features_dict['entropy_mixing']:.2f}")
                        st.metric("Components", f"{features_dict['n_components']}")

                    # Design rules
                    st.write("**📐 Design Rules:**")
                    r_ratio = features_dict['r_A_r_C']
                    if r_ratio < 0.35:
                        st.info("✓ r_A/r_C < 0.35 → Fluorite favorable")
                    elif r_ratio > 0.45:
                        st.info("✓ r_A/r_C > 0.45 → Spinel favorable")
                    else:
                        st.info("✓ r_A/r_C ~ 0.35-0.45 → Rock-salt favorable")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def concentration_sweep_mode(model, available_elements):
    """Analyze how concentration changes affect structure"""
    st.header("🔬 Concentration Sweep Analysis")
    st.info("See how changing one element's concentration affects the predicted structure")

    # Base composition
    st.subheader("Base Composition")
    n_components = st.slider("Number of cations:", 2, 5, 4, key="sweep_n")

    elements = []
    base_fractions = []

    cols = st.columns(n_components)
    for i in range(n_components):
        with cols[i]:
            elem = st.selectbox(
                f"Cation {i+1}:",
                available_elements,
                key=f"sweep_elem_{i}",
                index=min(i, len(available_elements)-1)
            )
            elements.append(elem)

            frac = st.number_input(
                f"Fraction:",
                min_value=0.0,
                max_value=1.0,
                value=1.0/n_components,
                step=0.05,
                key=f"sweep_frac_{i}",
                format="%.3f"
            )
            base_fractions.append(frac)

    vary_elem_idx = st.selectbox(
        "Vary concentration of:",
        range(len(elements)),
        format_func=lambda x: elements[x]
    )

    if st.button("🔍 Run Sweep", type="primary"):
        # Normalize base fractions
        total = sum(base_fractions)
        if total > 0:
            base_fractions = [f/total for f in base_fractions]

        # Create concentration sweep
        concentrations = np.linspace(0.05, 0.80, 30)
        predictions_data = []

        progress_bar = st.progress(0)
        for idx, conc in enumerate(concentrations):
            # Adjust fractions
            fractions = base_fractions.copy()
            fractions[vary_elem_idx] = conc

            # Renormalize others
            other_sum = sum([f for i, f in enumerate(fractions) if i != vary_elem_idx])
            if other_sum > 0:
                for i in range(len(fractions)):
                    if i != vary_elem_idx:
                        fractions[i] = fractions[i] * (1 - conc) / other_sum

            # Calculate features and predict
            try:
                features_dict = calculate_all_features(elements, fractions)
                features_array = features_to_array(features_dict)
                probabilities = model.predict_proba([features_array])[0]

                predictions_data.append({
                    'concentration': conc,
                    'Fluorite': probabilities[0],
                    'Rock-salt': probabilities[1],
                    'Spinel': probabilities[2],
                    'r_A_r_C': features_dict['r_A_r_C']
                })
            except:
                continue

            progress_bar.progress((idx + 1) / len(concentrations))

        if predictions_data:
            df = pd.DataFrame(predictions_data)

            # Plot probability vs concentration
            st.subheader("📈 Structure Probability vs Concentration")
            fig = go.Figure()

            for struct_name, color in [
                ('Fluorite', '#2196F3'),
                ('Rock-salt', '#FF9800'),
                ('Spinel', '#4CAF50')
            ]:
                fig.add_trace(go.Scatter(
                    x=df['concentration'],
                    y=df[struct_name],
                    mode='lines+markers',
                    name=struct_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6)
                ))

            fig.update_layout(
                title=f"Structure Probability vs {elements[vary_elem_idx]} Concentration",
                xaxis_title=f"{elements[vary_elem_idx]} Fraction",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # r_A/r_C plot
            st.subheader("📊 r_A/r_C Ratio vs Concentration")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df['concentration'],
                y=df['r_A_r_C'],
                mode='lines+markers',
                line=dict(color='#9C27B0', width=3),
                marker=dict(size=6)
            ))
            fig2.add_hline(y=0.35, line_dash="dash", line_color="blue", annotation_text="Fluorite")
            fig2.add_hline(y=0.45, line_dash="dash", line_color="green", annotation_text="Spinel")

            fig2.update_layout(
                xaxis_title=f"{elements[vary_elem_idx]} Fraction",
                yaxis_title="r_A/r_C Ratio",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Stability insights
            st.subheader("💡 Stability Insights")
            for struct_idx, struct_name in enumerate(['Fluorite', 'Rock-salt', 'Spinel']):
                max_prob_idx = df[struct_name].idxmax()
                max_prob = df.loc[max_prob_idx, struct_name]
                optimal_conc = df.loc[max_prob_idx, 'concentration']

                if max_prob > 0.6:
                    st.success(
                        f"**{struct_name}** most stable at "
                        f"{elements[vary_elem_idx]} = {optimal_conc:.3f} "
                        f"(probability: {max_prob*100:.1f}%)"
                    )


def batch_screening_mode(model, available_elements):
    """Screen multiple compositions"""
    st.header("⚗️ Batch Screening")
    st.info("Test many random compositions to find optimal structures")

    # Select elements
    selected_elements = st.multiselect(
        "Choose 3-5 elements to explore:",
        available_elements,
        default=available_elements[:4]
    )

    if len(selected_elements) < 3 or len(selected_elements) > 5:
        st.warning("Please select 3-5 elements")
        return

    target_structure = st.selectbox(
        "Filter by structure (optional):",
        ["All", "Fluorite", "Rock-salt", "Spinel"]
    )

    n_samples = st.slider("Number of compositions:", 50, 500, 100)

    if st.button("🚀 Run Screening", type="primary"):
        results = []
        progress_bar = st.progress(0)

        for i in range(n_samples):
            fractions = np.random.dirichlet(np.ones(len(selected_elements)))

            try:
                features_dict = calculate_all_features(selected_elements, fractions)
                features_array = features_to_array(features_dict)
                prediction = model.predict([features_array])[0]
                probabilities = model.predict_proba([features_array])[0]

                result = {
                    'Composition': ' + '.join([f"{f:.2f}{e}" for e, f in zip(selected_elements, fractions)]),
                    'Structure': STRUCTURES[prediction]['short'],
                    'Confidence': probabilities[prediction] * 100,
                    'r_A_r_C': features_dict['r_A_r_C'],
                    'ΔS_mix': features_dict['entropy_mixing']
                }
                results.append(result)
            except:
                continue

            progress_bar.progress((i + 1) / n_samples)

        if results:
            df = pd.DataFrame(results)

            # Filter
            if target_structure != "All":
                df = df[df['Structure'] == target_structure]

            df = df.sort_values('Confidence', ascending=False)

            # Summary
            st.subheader(f"📊 Results ({len(df)} compositions)")
            col1, col2, col3 = st.columns(3)

            all_df = pd.DataFrame(results)
            with col1:
                st.metric("Fluorite", f"{(all_df['Structure'] == 'Fluorite').sum()}")
            with col2:
                st.metric("Rock-salt", f"{(all_df['Structure'] == 'Rock-salt').sum()}")
            with col3:
                st.metric("Spinel", f"{(all_df['Structure'] == 'Spinel').sum()}")

            # Top results
            st.subheader("🏆 Top 10 Most Confident")
            st.dataframe(
                df.head(10)[['Composition', 'Structure', 'Confidence', 'r_A_r_C', 'ΔS_mix']].style.format({
                    'Confidence': '{:.1f}%',
                    'r_A_r_C': '{:.4f}',
                    'ΔS_mix': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            # Scatter plot
            fig = go.Figure()
            for struct_name, color in [
                ('Fluorite', '#2196F3'),
                ('Rock-salt', '#FF9800'),
                ('Spinel', '#4CAF50')
            ]:
                df_struct = all_df[all_df['Structure'] == struct_name]
                fig.add_trace(go.Scatter(
                    x=df_struct['r_A_r_C'],
                    y=df_struct['ΔS_mix'],
                    mode='markers',
                    name=struct_name,
                    marker=dict(color=color, size=8, opacity=0.6),
                    text=df_struct['Composition']
                ))

            fig.update_layout(
                title="Structure Map",
                xaxis_title="r_A/r_C",
                yaxis_title="ΔS_mix (J/mol·K)",
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "screening_results.csv",
                "text/csv"
            )


if __name__ == "__main__":
    main()
