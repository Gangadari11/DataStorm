import pandas as pd
import numpy as np

def prepare_features(data, is_training=True):
    """
    Prepare features for model training or prediction.
    Carefully avoids data leakage by not using target-related features.
    """
    # Create a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Convert date columns to datetime if they exist
    date_columns = ['year_month', 'agent_join_month', 'first_policy_sold_month']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=False, errors='coerce')
    
    # Time-based features
    # Tenure (months since joining)
    if all(col in df.columns for col in ['year_month', 'agent_join_month']):
        df['tenure_months'] = (df['year_month'] - df['agent_join_month']).dt.days / 30
    
    # Time since first sale (if available)
    if all(col in df.columns for col in ['year_month', 'first_policy_sold_month']):
        df['months_since_first_sale'] = np.where(
            df['first_policy_sold_month'].notna(),
            (df['year_month'] - df['first_policy_sold_month']).dt.days / 30,
            -1  # Placeholder for agents who haven't sold yet
        )
    
    # Time to first sale (for agents who have sold)
    if all(col in df.columns for col in ['first_policy_sold_month', 'agent_join_month']):
        if 'tenure_months' not in df.columns and 'year_month' in df.columns and 'agent_join_month' in df.columns:
            df['tenure_months'] = (df['year_month'] - df['agent_join_month']).dt.days / 30
            
        df['months_to_first_sale'] = np.where(
            df['first_policy_sold_month'].notna(),
            (df['first_policy_sold_month'] - df['agent_join_month']).dt.days / 30,
            df['tenure_months'] if 'tenure_months' in df.columns else -1  # For agents who haven't sold
        )
    
    # Seasonality features
    if 'year_month' in df.columns:
        df['month'] = df['year_month'].dt.month
        df['quarter'] = df['year_month'].dt.quarter
        df['is_q4'] = (df['quarter'] == 4).astype(int)  # Q4 often has different sales patterns
    
    # Activity-based features (avoiding direct use of target variables)
    
    # Proposal activity
    if 'unique_proposal' in df.columns:
        if 'tenure_months' in df.columns:
            df['proposal_intensity'] = df['unique_proposal'] / df['tenure_months'].replace(0, 1)
        
        if 'unique_proposals_last_7_days' in df.columns:
            df['proposal_recency'] = df['unique_proposals_last_7_days'] / df['unique_proposal'].replace(0, 1)
            
            if 'unique_proposals_last_15_days' in df.columns:
                df['proposal_trend_short'] = df['unique_proposals_last_7_days'] - df['unique_proposals_last_15_days']
                
                if 'unique_proposals_last_21_days' in df.columns:
                    df['proposal_trend_long'] = df['unique_proposals_last_15_days'] - df['unique_proposals_last_21_days']
                    df['proposal_acceleration'] = df['proposal_trend_short'] - df['proposal_trend_long']
    
    # Quotation activity
    if 'unique_quotations' in df.columns:
        if 'tenure_months' in df.columns:
            df['quotation_intensity'] = df['unique_quotations'] / df['tenure_months'].replace(0, 1)
        
        if 'unique_quotations_last_7_days' in df.columns:
            df['quotation_recency'] = df['unique_quotations_last_7_days'] / df['unique_quotations'].replace(0, 1)
            
            if 'unique_quotations_last_15_days' in df.columns:
                df['quotation_trend_short'] = df['unique_quotations_last_7_days'] - df['unique_quotations_last_15_days']
                
                if 'unique_quotations_last_21_days' in df.columns:
                    df['quotation_trend_long'] = df['unique_quotations_last_15_days'] - df['unique_quotations_last_21_days']
                    df['quotation_acceleration'] = df['quotation_trend_short'] - df['quotation_trend_long']
    
    # Customer activity
    if 'unique_customers' in df.columns:
        if 'tenure_months' in df.columns:
            df['customer_intensity'] = df['unique_customers'] / df['tenure_months'].replace(0, 1)
        
        if 'unique_customers_last_7_days' in df.columns:
            df['customer_recency'] = df['unique_customers_last_7_days'] / df['unique_customers'].replace(0, 1)
            
            if 'unique_customers_last_15_days' in df.columns:
                df['customer_trend_short'] = df['unique_customers_last_7_days'] - df['unique_customers_last_15_days']
                
                if 'unique_customers_last_21_days' in df.columns:
                    df['customer_trend_long'] = df['unique_customers_last_15_days'] - df['unique_customers_last_21_days']
                    df['customer_acceleration'] = df['customer_trend_short'] - df['customer_trend_long']
    
    # Conversion efficiency (without using target variables)
    if all(col in df.columns for col in ['unique_quotations', 'unique_proposal']):
        df['proposal_to_quotation_ratio'] = df['unique_quotations'] / df['unique_proposal'].replace(0, 1)
    
    if all(col in df.columns for col in ['unique_proposal', 'unique_customers']):
        df['customer_to_proposal_ratio'] = df['unique_proposal'] / df['unique_customers'].replace(0, 1)
    
    # Activity diversity
    if all(col in df.columns for col in ['unique_quotations', 'unique_proposal']):
        df['activity_diversity'] = df['unique_quotations'] / (df['unique_proposal'] + 0.1)
    
    # Agent characteristics
    if 'agent_age' in df.columns:
        df['is_young_agent'] = (df['agent_age'] < 30).astype(int)
        df['is_senior_agent'] = (df['agent_age'] > 50).astype(int)
    
    if 'tenure_months' in df.columns:
        df['is_new_agent'] = (df['tenure_months'] < 6).astype(int)
        df['is_experienced_agent'] = (df['tenure_months'] > 24).astype(int)
    
    # Interaction features
    if all(col in df.columns for col in ['agent_age', 'tenure_months']):
        df['age_tenure_interaction'] = df['agent_age'] * df['tenure_months']
    
    if all(col in df.columns for col in ['unique_proposal', 'unique_quotations']):
        df['proposal_quotation_interaction'] = df['unique_proposal'] * df['unique_quotations']
    
    # Cash payment preference (might indicate customer demographics)
    if all(col in df.columns for col in ['number_of_cash_payment_policies', 'number_of_policy_holders']):
        df['cash_payment_ratio'] = df['number_of_cash_payment_policies'] / df['number_of_policy_holders'].replace(0, 1)
    
    # Drop columns that would cause data leakage or aren't useful for prediction
    cols_to_drop = [
        'year_month', 'agent_join_month', 'first_policy_sold_month',  # Date columns
        'new_policy_count', 'ANBP_value', 'net_income'  # Target-related columns
    ]
    
    # Only drop columns that exist in the dataframe
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if not is_training:
        # For test data, also drop the target column if it exists
        if 'target_column' in df.columns:
            cols_to_drop.append('target_column')
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    for col in cat_cols:
        if col not in ['row_id', 'agent_code', 'agent_name']:  # Skip ID columns
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df
