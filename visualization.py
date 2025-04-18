import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime
import re
import numpy as np
import json
import traceback

def detect_column_types(df):
    """
    Detect the types of columns in the DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary with lists of column names by type
    """
    try:
        if df.empty:
            return {
                'numerical': [],
                'categorical': [],
                'datetime': [],
                'text': []
            }
        
        column_types = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }
        
        for column in df.columns:
            try:
                # Skip columns with all null values
                if df[column].isna().all():
                    continue
                
                # First, check for any potential error-causing values (mixed types, etc.)
                try:
                    # Try to convert to JSON-safe format - this will catch many potential issues
                    sample_value = df[column].iloc[0] if len(df) > 0 else None
                    if isinstance(sample_value, (list, dict, set, tuple)) or str(type(sample_value)).find("<class '") >= 0:
                        # Complex or custom object types get treated as text
                        column_types['text'].append(column)
                        continue
                except Exception:
                    # Any error indicates a complex type we should handle carefully
                    column_types['text'].append(column)
                    continue
                
                # Check if column contains unhashable types (like lists, dicts)
                has_unhashable_types = False
                try:
                    # A simple way to check is to try counting unique values
                    df[column].nunique()
                except (TypeError, ValueError):
                    # If TypeError/ValueError occurs, it likely contains unhashable types
                    has_unhashable_types = True
                
                # If column has unhashable types, treat as text/categorical
                if has_unhashable_types:
                    # Convert to string and check length
                    try:
                        avg_str_len = df[column].astype(str).str.len().mean()
                        if avg_str_len > 50:
                            column_types['text'].append(column)
                        else:
                            column_types['categorical'].append(column)
                    except:
                        # If even string conversion fails, default to text
                        column_types['text'].append(column)
                    continue
                    
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[column], errors='raise')
                    column_types['datetime'].append(column)
                    continue
                except:
                    pass
                
                # Check data type
                if pd.api.types.is_numeric_dtype(df[column]):
                    column_types['numerical'].append(column)
                else:
                    # For non-numeric columns, check unique value count safely
                    try:
                        unique_count = df[column].nunique()
                        if unique_count <= max(5, len(df) * 0.2):  # Consider as categorical if few unique values
                            column_types['categorical'].append(column)
                        else:
                            # Check average string length to differentiate between short categorical and long text
                            avg_len = df[column].astype(str).str.len().mean()
                            if avg_len > 50:
                                column_types['text'].append(column)
                            else:
                                column_types['categorical'].append(column)
                    except:
                        # If we can't count unique values, default to categorical
                        column_types['categorical'].append(column)
            except Exception as e:
                # If any column processing fails completely, add to text as safest option
                st.warning(f"Error processing column '{column}': {str(e)}")
                column_types['text'].append(column)
                    
        return column_types
    except Exception as e:
        # Complete failure fallback - return empty categories
        st.error(f"Failed to detect column types: {str(e)}")
        return {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }

def create_visualization(df, metadata, column_types):
    """
    Create appropriate visualizations based on query results and metadata.
    
    Args:
        df: pandas DataFrame with query results
        metadata: Dictionary with query metadata
        column_types: Dictionary with column types
    
    Returns:
        List of Plotly figures
    """
    if df.empty:
        st.warning("No data available for visualization")
        return []
    
    # Handle special case for COUNT(*) results (single cell results)
    if df.shape == (1, 1):
        # This is handled in app.py directly now, so we'll return an empty list here
        # to avoid duplicate visualizations
        return []
    
    figures = []
    viz_type = metadata.get('visualization_type', 'table')
    query_type = metadata.get('query_type', 'simple')
    
    # Check for column names that might cause issues and rename them
    renamed_df = df.copy()
    column_mapping = {}
    for col in renamed_df.columns:
        # Check for SQL function names in columns that commonly cause issues
        if any(pattern in col.lower() for pattern in ['count(', 'sum(', 'avg(', 'min(', 'max(']):
            # Extract function name from the column
            match = re.search(r'(\w+)\(', col.lower())
            if match:
                function_name = match.group(1).capitalize()
                # Create a more friendly name
                if '*' in col:
                    new_name = f"{function_name} of Rows"
                else:
                    # Extract the column name from inside the function
                    field_match = re.search(r'\((.*?)\)', col)
                    field = field_match.group(1) if field_match else "value"
                    field = field.replace('*', 'all').replace('"', '').replace("'", "")
                    new_name = f"{function_name} of {field}"
                
                # Store the mapping and rename the column
                column_mapping[col] = new_name
                renamed_df = renamed_df.rename(columns={col: new_name})
    
    # If we renamed any columns, update the metadata and column_types accordingly
    if column_mapping:
        # Update metadata
        for key in ['x_axis', 'y_axis', 'group_by']:
            if key in metadata and metadata[key] in column_mapping:
                metadata[key] = column_mapping[metadata[key]]
        
        # Update column_types
        for category in column_types:
            updated_cols = []
            for col in column_types[category]:
                if col in column_mapping:
                    updated_cols.append(column_mapping[col])
                else:
                    updated_cols.append(col)
            column_types[category] = updated_cols
        
        # Use the renamed dataframe for all visualizations
        df = renamed_df
    
    # Create a fallback table visualization regardless of chart generation results
    # This ensures we always have at least one visualization
    try:
        # Create a clean version of the dataframe that handles None and NaN values
        # to prevent JavaScript errors in the visualization
        df_clean = df.copy()
        for col in df_clean.columns:
            # Replace NaN values with appropriate placeholders based on column type
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0)
            elif pd.api.types.is_datetime64_dtype(df_clean[col]) or pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(pd.Timestamp('now'))
            else:
                df_clean[col] = df_clean[col].fillna("") 
        
        # Create table with cleaned data
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(df_clean.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df_clean[col].astype(str).tolist() for col in df_clean.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig_table.update_layout(title="Data Table")
        figures.append(fig_table)
    except Exception as e:
        st.warning(f"Error creating table visualization: {str(e)}")
    
    # This code for report mode is kept for backward compatibility
    # but we no longer return early to allow for consistent visualization behavior
    # between tables mode and reports mode
    if query_type == 'report':
        try:
            # For reports, we need to infer appropriate columns without relying on metadata
            # We'll create multiple visualizations based on the data types
            
            # Generic bar chart for any report with categorical and numerical data
            if column_types['categorical'] and column_types['numerical']:
                # For each numerical column, create a bar chart with the first categorical column
                for y_col in column_types['numerical'][:2]:  # Use first 2 numerical columns at most
                    x_col = column_types['categorical'][0]  # Use the first categorical column
                    
                    # Verify columns exist in dataframe
                    if x_col not in df.columns or y_col not in df.columns:
                        continue
                        
                    # Create bar chart
                    fig = px.bar(df, x=x_col, y=y_col,
                               title=f"{y_col} by {x_col}",
                               template="plotly_white")
                    figures.append(fig)
            
            # Time series chart if we have datetime columns
            if column_types['datetime'] and column_types['numerical']:
                x_col = column_types['datetime'][0]
                # Verify column exists
                if x_col in df.columns:
                    for y_col in column_types['numerical'][:2]:  # Use first 2 numerical columns at most
                        if y_col not in df.columns:
                            continue
                            
                        # Sort by date
                        df_sorted = df.sort_values(by=x_col)
                        fig = px.line(df_sorted, x=x_col, y=y_col,
                                     title=f"{y_col} over time",
                                     template="plotly_white")
                        figures.append(fig)
            
            # Add a pie chart for the first categorical and numerical columns
            if column_types['categorical'] and column_types['numerical']:
                x_col = column_types['categorical'][0]
                y_col = column_types['numerical'][0]
                
                # Verify columns exist
                if x_col in df.columns and y_col in df.columns:
                    # Only create pie chart if not too many categories
                    if df[x_col].nunique() <= 10:
                        fig = px.pie(df, names=x_col, values=y_col,
                                    title=f"Distribution of {y_col} by {x_col}")
                        figures.append(fig)
        except Exception as e:
            st.warning(f"Error creating report visualizations: {str(e)}")
            # We'll still have the table visualization as a fallback
        
        # We don't return early anymore, let this continue to the standard visualization processing
        # so reports get the same visualizations as tables
    
    # Standard visualization processing for natural language queries
    try:
        # Determine columns to use for visualization
        x_col = metadata.get('x_axis')
        y_col = metadata.get('y_axis')
        group_col = metadata.get('group_by')
        
        # If columns are not specified in metadata, try to infer them
        if not x_col and column_types['categorical']:
            x_col = column_types['categorical'][0]
        if not y_col and column_types['numerical']:
            y_col = column_types['numerical'][0]
        if not group_col and len(column_types['categorical']) > 1:
            candidates = [col for col in column_types['categorical'] if col != x_col]
            if candidates:
                group_col = candidates[0]
                
        # Check if inferred columns exist in the dataframe
        if x_col and x_col not in df.columns:
            # Don't show warning - this is expected in some cases, especially with renamed columns
            x_col = None
        if y_col and y_col not in df.columns:
            # Don't show warning - this is expected in some cases, especially with renamed columns
            y_col = None
                
        # If columns weren't found, try to find fallback columns
        if not x_col and df.columns.size > 0:
            x_col = df.columns[0]  # Use first column as fallback
        if not y_col and df.columns.size > 1:
            # Find a numeric column as fallback
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    y_col = col
                    break
        
        # Bar chart
        if viz_type == 'bar' and x_col and y_col:
            if group_col and group_col in df.columns:
                fig = px.bar(df, x=x_col, y=y_col, color=group_col, 
                            title=f"{y_col} by {x_col}", 
                            template="plotly_white")
            else:
                fig = px.bar(df, x=x_col, y=y_col,
                            title=f"{y_col} by {x_col}", 
                            template="plotly_white")
            figures.append(fig)
            
        # Line chart
        elif viz_type == 'line' and x_col and y_col:
            if x_col in column_types['datetime']:
                # Ensure datetime is sorted
                df_sorted = df.sort_values(by=x_col)
                if group_col and group_col in df.columns:
                    fig = px.line(df_sorted, x=x_col, y=y_col, color=group_col,
                                title=f"{y_col} over time", 
                                template="plotly_white")
                else:
                    fig = px.line(df_sorted, x=x_col, y=y_col,
                                title=f"{y_col} over time", 
                                template="plotly_white")
            else:
                if group_col and group_col in df.columns:
                    fig = px.line(df, x=x_col, y=y_col, color=group_col,
                                title=f"{y_col} by {x_col}", 
                                template="plotly_white")
                else:
                    fig = px.line(df, x=x_col, y=y_col,
                                title=f"{y_col} by {x_col}", 
                                template="plotly_white")
            figures.append(fig)
            
        # Pie chart
        elif viz_type == 'pie' and x_col and y_col:
            fig = px.pie(df, names=x_col, values=y_col,
                        title=f"Distribution of {y_col} by {x_col}", 
                        template="plotly_white")
            figures.append(fig)
            
        # Scatter plot
        elif viz_type == 'scatter' and len(column_types['numerical']) >= 2:
            if not x_col or x_col not in column_types['numerical']:
                x_col = column_types['numerical'][0]
            if not y_col or y_col not in column_types['numerical']:
                y_col = column_types['numerical'][1] if len(column_types['numerical']) > 1 else column_types['numerical'][0]
                
            if group_col and group_col in df.columns:
                fig = px.scatter(df, x=x_col, y=y_col, color=group_col,
                                title=f"Relationship between {x_col} and {y_col}", 
                                template="plotly_white")
            else:
                fig = px.scatter(df, x=x_col, y=y_col,
                                title=f"Relationship between {x_col} and {y_col}", 
                                template="plotly_white")
            figures.append(fig)
            
        # Histogram
        elif viz_type == 'histogram' and column_types['numerical']:
            if not x_col or x_col not in column_types['numerical']:
                x_col = column_types['numerical'][0]
                
            if group_col and group_col in df.columns:
                fig = px.histogram(df, x=x_col, color=group_col,
                                  title=f"Distribution of {x_col}", 
                                  template="plotly_white")
            else:
                fig = px.histogram(df, x=x_col,
                                  title=f"Distribution of {x_col}", 
                                  template="plotly_white")
            figures.append(fig)
            
        # Box plot
        elif viz_type == 'box' and column_types['numerical'] and column_types['categorical']:
            if not x_col or x_col not in column_types['categorical']:
                x_col = column_types['categorical'][0]
            if not y_col or y_col not in column_types['numerical']:
                y_col = column_types['numerical'][0]
                
            fig = px.box(df, x=x_col, y=y_col,
                        title=f"Distribution of {y_col} by {x_col}", 
                        template="plotly_white")
            figures.append(fig)
        
        # For aggregate queries, add a summary visualization
        if query_type == 'aggregate' and x_col and y_col:
            agg_fig = px.bar(df, x=x_col, y=y_col, 
                           title=f"Summary of {y_col} by {x_col}",
                           template="plotly_white")
            figures.append(agg_fig)
    except Exception as e:
        st.warning(f"Error creating visualizations: {str(e)}")
        # We'll still have the table visualization as a fallback
    
    return figures

def create_advanced_visualization(df, viz_type, columns, settings=None):
    """
    Create advanced visualizations with customizable settings.
    
    Args:
        df: pandas DataFrame with data
        viz_type: Type of visualization to create
        columns: Dictionary of column names to use for different axes/metrics
        settings: Dictionary of visualization settings and customizations
    
    Returns:
        Plotly figure object or None if visualization could not be created
    """
    if settings is None:
        settings = {}
    
    # Print debug info for troubleshooting
    if st.session_state.get('debug_mode', False):
        st.write(f"Viz type: {viz_type}")
        st.write(f"Columns: {columns}")
        st.write(f"Settings: {settings}")
        st.write(f"DF columns: {df.columns.tolist()}")
    
    # Validate input parameters
    if df.empty:
        if st.session_state.get('debug_mode', False):
            st.warning("Cannot create visualization: DataFrame is empty")
        return None
    
    if not isinstance(columns, dict):
        if st.session_state.get('debug_mode', False):
            st.warning(f"Invalid columns format: expected dict, got {type(columns)}")
        # Try to convert to dict if possible
        if isinstance(columns, str):
            try:
                columns = {"value": columns}
            except:
                return None
        else:
            return None
    
    # Create a clean copy of the dataframe with null values handled
    # to prevent JavaScript errors in visualizations
    try:
        df_clean = df.copy()
        for col in df_clean.columns:
            # Replace NaN values with appropriate placeholders based on column type
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0)
            elif pd.api.types.is_datetime64_dtype(df_clean[col]) or pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(pd.Timestamp('now'))
            else:
                df_clean[col] = df_clean[col].fillna("")
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error cleaning DataFrame: {str(e)}")
        return None
    
    # Use the cleaned dataframe for all visualizations
    df = df_clean
    
    # Default settings
    color_theme = settings.get('color_theme', 'plotly')
    title = settings.get('title', '')
    height = settings.get('height', 600)
    width = settings.get('width', None)  # Auto width by default
    show_legend = settings.get('show_legend', True)
    
    # Theme/templates mapping
    theme_mapping = {
        'default': 'plotly',
        'light': 'plotly_white',
        'dark': 'plotly_dark',
        'minimal': 'seaborn',
        'business': 'presentation',
        'vibrant': 'ggplot2'
    }
    template = theme_mapping.get(color_theme, 'plotly_white')
    
    # Common layout settings
    layout_settings = {
        'title': title,
        'height': height,
        'width': width,
        'template': template,
        'showlegend': show_legend
    }
    
    # Add title margin if title is provided
    if title:
        layout_settings['margin'] = dict(t=60, l=40, r=40, b=40)
    
    # Basic layout config with theme
    layout_config = {'layout': layout_settings}
    
    # Create visualization based on type
    if viz_type == 'advanced_table':
        # Get column formatting settings
        col_formats = settings.get('column_formats', {})
        
        # Create advanced table with custom formatting and sorting
        header_colors = settings.get('header_colors', ['paleturquoise', 'white'])
        cell_colors = settings.get('cell_colors', ['lavender', 'white'])
        
        # Format cell values according to column formats
        formatted_values = []
        for col in df.columns:
            if col in col_formats:
                format_type = col_formats[col].get('type', 'auto')
                
                if format_type == 'number':
                    # Format numbers with specified precision
                    precision = col_formats[col].get('precision', 2)
                    formatted_values.append([f"{val:.{precision}f}" if pd.notna(val) else "" for val in df[col]])
                elif format_type == 'percentage':
                    # Format as percentages
                    precision = col_formats[col].get('precision', 1)
                    formatted_values.append([f"{val*100:.{precision}f}%" if pd.notna(val) else "" for val in df[col]])
                elif format_type == 'date':
                    # Format as dates with specified format
                    date_format = col_formats[col].get('format', '%Y-%m-%d')
                    formatted_values.append([val.strftime(date_format) if pd.notna(val) else "" for val in pd.to_datetime(df[col], errors='coerce')])
                elif format_type == 'currency':
                    # Format as currency
                    currency_symbol = col_formats[col].get('currency_symbol', '$')
                    precision = col_formats[col].get('precision', 2)
                    formatted_values.append([f"{currency_symbol}{val:.{precision}f}" if pd.notna(val) else "" for val in df[col]])
                else:
                    # Default formatting
                    formatted_values.append(df[col].astype(str).tolist())
            else:
                # Default formatting for columns without specific format
                formatted_values.append(df[col].astype(str).tolist())
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color=header_colors[0],
                align='left',
                font=dict(size=13)
            ),
            cells=dict(
                values=formatted_values,
                fill_color=[cell_colors[0] if i%2==0 else cell_colors[1] for i in range(len(df))],
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(**layout_settings)
        return fig
    
    elif viz_type == 'treemap':
        # Create treemap visualization with robust error handling
        try:
            x_col = columns.get('path')
            values_col = columns.get('values')
            color_col = columns.get('color')
            
            # Convert any unhashable types to strings to avoid errors
            if isinstance(x_col, list):
                path_data = []
                path_data.append(px.Constant("Total"))
                
                for col in x_col:
                    # Check if column contains unhashable types
                    try:
                        df[col].nunique()  # This will fail if column has unhashable values
                        path_data.append(df[col])
                    except TypeError:
                        # Convert to string if unhashable
                        path_data.append(df[col].astype(str))
                
                path = path_data
            else:
                # Single column path
                try:
                    df[x_col].nunique()  # Check if hashable
                    path = [px.Constant("Total"), df[x_col]]
                except TypeError:
                    # Convert to string if unhashable
                    path = [px.Constant("Total"), df[x_col].astype(str)]
            
            # Handle color column if it contains unhashable types
            color_data = None
            if color_col:
                try:
                    df[color_col].nunique()  # Check if hashable
                    color_data = df[color_col]
                except TypeError:
                    # If unhashable, either convert to string or use a default color
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        # For numeric columns with unhashable elements, use the first numeric column instead
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            color_data = df[numeric_cols[0]]
                    else:
                        # For non-numeric, convert to string
                        color_data = df[color_col].astype(str)
            
            fig = px.treemap(
                df, 
                path=path,
                values=values_col if values_col else None,
                color=color_data,
                hover_data=settings.get('hover_data', None),
                color_continuous_scale=settings.get('color_scale', 'RdBu'),
                **layout_config
            )
        except Exception as e:
            st.warning(f"Error creating treemap: {str(e)}")
            # Create a simple fallback visualization
            fig = go.Figure(go.Treemap(
                labels=["Error", "Could not create treemap", "Try a different visualization"],
                parents=["", "Error", "Error"]
            ))
            fig.update_layout(**layout_settings)
        return fig
    
    elif viz_type == 'sunburst':
        # Create sunburst visualization for hierarchical data with robust error handling
        try:
            path_cols = columns.get('path', [])
            values_col = columns.get('values')
            color_col = columns.get('color')
            
            if not isinstance(path_cols, list):
                path_cols = [path_cols]
                
            # Build path with error handling for unhashable types
            path_data = [px.Constant("Total")]
            
            for col in path_cols:
                # Check if column contains unhashable types
                try:
                    df[col].nunique()  # This will fail if column has unhashable values
                    path_data.append(df[col])
                except TypeError:
                    # Convert to string if unhashable
                    path_data.append(df[col].astype(str))
            
            # Handle color column if it contains unhashable types
            color_data = None
            if color_col:
                try:
                    df[color_col].nunique()  # Check if hashable
                    color_data = df[color_col]
                except TypeError:
                    # If unhashable, either convert to string or use a default color
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        # For numeric columns with unhashable elements, use the first numeric column instead
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            color_data = df[numeric_cols[0]]
                    else:
                        # For non-numeric, convert to string
                        color_data = df[color_col].astype(str)
            
            fig = px.sunburst(
                df,
                path=path_data,
                values=values_col,
                color=color_data,
                **layout_config
            )
        except Exception as e:
            st.warning(f"Error creating sunburst chart: {str(e)}")
            # Create a simple fallback visualization
            fig = go.Figure(go.Sunburst(
                labels=["Error", "Could not create sunburst", "Try a different visualization"],
                parents=["", "Error", "Error"]
            ))
            fig.update_layout(**layout_settings)
        return fig
    
    elif viz_type == 'heatmap':
        # Create heatmap visualization
        x_col = columns.get('x')
        y_col = columns.get('y')
        z_col = columns.get('z')
        
        # If z_col is provided, pivot the data
        if z_col:
            try:
                pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
                z_data = pivot_df.values
                x_labels = pivot_df.columns.tolist()
                y_labels = pivot_df.index.tolist()
            except Exception as e:
                st.warning(f"Error pivoting data for heatmap: {str(e)}")
                # Fallback to basic correlation heatmap
                corr_df = df.select_dtypes(include=['number']).corr()
                z_data = corr_df.values
                x_labels = corr_df.columns.tolist()
                y_labels = corr_df.index.tolist()
        else:
            # Create correlation heatmap by default
            corr_df = df.select_dtypes(include=['number']).corr()
            z_data = corr_df.values
            x_labels = corr_df.columns.tolist()
            y_labels = corr_df.index.tolist()
        
        # Create heatmap with customizations
        color_scale = settings.get('color_scale', 'RdBu_r')
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=color_scale,
            zmin=settings.get('z_min', -1),
            zmax=settings.get('z_max', 1),
            colorbar=dict(title=settings.get('colorbar_title', 'Value'))
        ))
        
        # Add annotations if requested
        if settings.get('show_annotations', False):
            annotations = []
            for i, y_label in enumerate(y_labels):
                for j, x_label in enumerate(x_labels):
                    annotations.append(dict(
                        x=x_label,
                        y=y_label,
                        text=str(round(z_data[i][j], 2)),
                        showarrow=False,
                        font=dict(color='black' if abs(z_data[i][j]) < 0.5 else 'white')
                    ))
            fig.update_layout(annotations=annotations)
        
        fig.update_layout(**layout_settings)
        return fig
    
    elif viz_type == 'bubble':
        # Create bubble chart with size and color dimensions
        x_col = columns.get('x')
        y_col = columns.get('y')
        size_col = columns.get('size')
        color_col = columns.get('color')
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            hover_name=settings.get('hover_name', None),
            hover_data=settings.get('hover_data', None),
            size_max=settings.get('size_max', 40),
            opacity=settings.get('opacity', 0.7),
            **layout_config
        )
        return fig
    
    elif viz_type == 'candlestick':
        # Create candlestick chart for financial data
        date_col = columns.get('date')
        open_col = columns.get('open')
        high_col = columns.get('high')
        low_col = columns.get('low')
        close_col = columns.get('close')
        
        fig = go.Figure(data=[go.Candlestick(
            x=df[date_col],
            open=df[open_col], 
            high=df[high_col],
            low=df[low_col], 
            close=df[close_col]
        )])
        
        fig.update_layout(**layout_settings)
        return fig
    
    elif viz_type == 'waterfall':
        # Create waterfall chart to show cumulative effect
        x_col = columns.get('x')
        y_col = columns.get('y')
        
        # Adjust colors based on values being positive or negative
        measure = ['relative'] * len(df)
        
        # Set total measure if requested
        if settings.get('show_total', True) and len(df) > 0:
            df = df.copy()
            total_name = settings.get('total_name', 'Total')
            total_value = df[y_col].sum()
            df.loc[len(df)] = [total_name, total_value]
            measure.append('total')
        
        fig = go.Figure(go.Waterfall(
            name="Waterfall",
            orientation="v",
            measure=measure,
            x=df[x_col],
            y=df[y_col],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": settings.get('decreasing_color', "red")}},
            increasing={"marker": {"color": settings.get('increasing_color', "green")}},
            totals={"marker": {"color": settings.get('total_color', "blue")}}
        ))
        
        fig.update_layout(**layout_settings)
        return fig
    
    elif viz_type == 'parallel_coordinates':
        # Create parallel coordinates plot for multi-dimensional data
        columns_to_use = columns.get('dimensions', df.select_dtypes(include=['number']).columns.tolist())
        color_col = columns.get('color')
        
        fig = px.parallel_coordinates(
            df,
            dimensions=columns_to_use,
            color=df[color_col] if color_col else None,
            color_continuous_scale=settings.get('color_scale', 'Viridis'),
            **layout_config
        )
        return fig
    
    elif viz_type == 'radar':
        # Create radar chart (polar chart)
        categories = columns.get('categories')
        values = columns.get('values')
        
        if isinstance(categories, list) and isinstance(values, list):
            fig = go.Figure()
            
            # Add multiple traces for comparison if needed
            for i, value_col in enumerate(values):
                fig.add_trace(go.Scatterpolar(
                    r=df[value_col].tolist(),
                    theta=df[categories].tolist(),
                    fill='toself',
                    name=value_col
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, df[values].max().max() * 1.1]
                    )
                ),
                **layout_settings
            )
            return fig
    
    elif viz_type == 'density_contour':
        # Create density contour plot
        x_col = columns.get('x')
        y_col = columns.get('y')
        
        fig = px.density_contour(
            df, 
            x=x_col, 
            y=y_col,
            marginal_x=settings.get('marginal_x', 'histogram'),
            marginal_y=settings.get('marginal_y', 'histogram'),
            **layout_config
        )
        return fig
    
    elif viz_type == 'gantt':
        # Create Gantt chart for project timelines
        task_col = columns.get('task')
        start_col = columns.get('start')
        end_col = columns.get('end')
        
        fig = px.timeline(
            df,
            x_start=start_col,
            x_end=end_col,
            y=task_col,
            color=columns.get('color'),
            **layout_config
        )
        
        # Adjust layout for Gantt chart
        fig.update_yaxes(autorange="reversed")
        
        return fig
    
    elif viz_type == 'multi_chart':
        # Create visualization with multiple subplots
        subplot_type = settings.get('subplot_type', 'grid')
        rows = settings.get('rows', 2)
        cols = settings.get('cols', 1)
        
        subplot_titles = settings.get('subplot_titles', None)
        shared_xaxes = settings.get('shared_xaxes', False)
        shared_yaxes = settings.get('shared_yaxes', False)
        
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            subplot_titles=subplot_titles,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            specs=settings.get('specs', None)
        )
        
        # Default layout
        fig.update_layout(**layout_settings)
        
        return fig
        
    elif viz_type == 'distribution':
        # Create distribution visualization using distplot or histograms
        value_col = columns.get('value')
        group_col = columns.get('group')
        
        # Use figure factory for advanced distribution or fallback to histogram
        try:
            if not group_col:
                # Simple distribution for one column
                fig = ff.create_distplot(
                    [df[value_col].dropna()], 
                    [value_col],
                    bin_size=settings.get('bin_size', None),
                    show_hist=settings.get('show_hist', True),
                    show_curve=settings.get('show_curve', True),
                    show_rug=settings.get('show_rug', False)
                )
            else:
                # Create distribution by group
                groups = df[group_col].unique()
                hist_data = []
                group_labels = []
                
                for group in groups:
                    group_data = df[df[group_col] == group][value_col].dropna()
                    if len(group_data) > 0:
                        hist_data.append(group_data)
                        group_labels.append(str(group))
                
                if hist_data:
                    fig = ff.create_distplot(
                        hist_data, 
                        group_labels,
                        bin_size=settings.get('bin_size', None),
                        show_hist=settings.get('show_hist', True),
                        show_curve=settings.get('show_curve', True),
                        show_rug=settings.get('show_rug', False)
                    )
                else:
                    # Fallback if no valid groups
                    fig = px.histogram(
                        df, 
                        x=value_col,
                        marginal=settings.get('marginal', 'box'),
                        **layout_config
                    )
        except Exception as e:
            # Fallback to basic histogram
            st.warning(f"Error creating distribution plot: {str(e)}")
            fig = px.histogram(
                df, 
                x=value_col,
                color=group_col if group_col else None,
                marginal=settings.get('marginal', 'box'),
                **layout_config
            )
        
        fig.update_layout(**layout_settings)
        return fig
    
    # Return None if visualization type is not supported
    return None

def suggest_visualizations(df, query_text):
    """
    Suggest appropriate visualizations based on the data and query.
    
    Args:
        df: pandas DataFrame with query results
        query_text: Natural language query
    
    Returns:
        Dictionary with visualization suggestions
    """
    if df.empty:
        return {"message": "No data available for visualization"}
    
    column_types = detect_column_types(df)
    
    # Basic suggestions based on column types
    suggestions = []
    
    # Add basic suggestions based on column types
    if column_types['numerical'] and column_types['categorical']:
        suggestions.append({
            "type": "bar_chart",
            "description": f"Bar chart showing {column_types['numerical'][0]} by {column_types['categorical'][0]}",
            "columns": {
                "x": column_types['categorical'][0],
                "y": column_types['numerical'][0]
            }
        })
        
    if len(column_types['numerical']) >= 2:
        suggestions.append({
            "type": "scatter_plot",
            "description": f"Scatter plot showing relationship between {column_types['numerical'][0]} and {column_types['numerical'][1]}",
            "columns": {
                "x": column_types['numerical'][0],
                "y": column_types['numerical'][1]
            }
        })
        
        # Add bubble chart suggestion if there's a third numerical column
        if len(column_types['numerical']) >= 3:
            suggestions.append({
                "type": "bubble_chart",
                "description": f"Bubble chart showing {column_types['numerical'][0]} vs {column_types['numerical'][1]} with {column_types['numerical'][2]} as size",
                "columns": {
                    "x": column_types['numerical'][0],
                    "y": column_types['numerical'][1],
                    "size": column_types['numerical'][2]
                }
            })
        
        # Add heatmap for correlation analysis
        suggestions.append({
            "type": "heatmap",
            "description": "Correlation heatmap for numerical variables",
            "columns": {
                "numerical": column_types['numerical']
            }
        })
        
    if column_types['datetime'] and column_types['numerical']:
        suggestions.append({
            "type": "line_chart",
            "description": f"Line chart showing {column_types['numerical'][0]} over time ({column_types['datetime'][0]})",
            "columns": {
                "x": column_types['datetime'][0],
                "y": column_types['numerical'][0]
            }
        })
        
    if column_types['categorical'] and len(df[column_types['categorical'][0]].unique()) <= 10:
        for num_col in column_types['numerical']:
            suggestions.append({
                "type": "pie_chart",
                "description": f"Pie chart showing distribution of {num_col} by {column_types['categorical'][0]}",
                "columns": {
                    "names": column_types['categorical'][0],
                    "values": num_col
                }
            })
    
    # Add histogram suggestion for numerical columns
    if column_types['numerical']:
        suggestions.append({
            "type": "histogram",
            "description": f"Histogram showing distribution of {column_types['numerical'][0]}",
            "columns": {
                "x": column_types['numerical'][0]
            }
        })
    
    # Add box plot suggestion
    if column_types['numerical'] and column_types['categorical']:
        suggestions.append({
            "type": "box_plot",
            "description": f"Box plot showing distribution of {column_types['numerical'][0]} by {column_types['categorical'][0]}",
            "columns": {
                "x": column_types['categorical'][0],
                "y": column_types['numerical'][0]
            }
        })
    
    # Add treemap suggestion if appropriate
    if len(column_types['categorical']) >= 2 and column_types['numerical']:
        suggestions.append({
            "type": "treemap",
            "description": f"Treemap showing {column_types['numerical'][0]} by {', '.join(column_types['categorical'][:2])}",
            "columns": {
                "path": column_types['categorical'][:2],
                "values": column_types['numerical'][0]
            }
        })
    
    # Add radar chart suggestion for multi-dimensional comparison
    if len(column_types['numerical']) >= 3 and column_types['categorical']:
        suggestions.append({
            "type": "radar_chart",
            "description": f"Radar chart comparing multiple metrics across {column_types['categorical'][0]}",
            "columns": {
                "categories": column_types['categorical'][0],
                "values": column_types['numerical'][:3]
            }
        })
            
    return suggestions

def apply_custom_theme(fig, theme_name="default"):
    """
    Apply a custom theme to a plotly figure.
    
    Args:
        fig: Plotly figure object
        theme_name: Name of the theme to apply
    
    Returns:
        Updated Plotly figure
    """
    themes = {
        "default": {
            "bgcolor": "white",
            "textcolor": "#444",
            "gridcolor": "#eee",
            "colorscale": "Viridis"
        },
        "dark": {
            "bgcolor": "#222",
            "textcolor": "white",
            "gridcolor": "#555",
            "colorscale": "Plasma"
        },
        "corporate": {
            "bgcolor": "white",
            "textcolor": "#505050",
            "gridcolor": "#e0e0e0",
            "colorscale": "Blues"
        },
        "modern": {
            "bgcolor": "#f9f9f9",
            "textcolor": "#333",
            "gridcolor": "#ddd",
            "colorscale": "Teal"
        },
        "minimal": {
            "bgcolor": "white",
            "textcolor": "#333",
            "gridcolor": "#f0f0f0",
            "colorscale": "Greys"
        }
    }
    
    # Use default theme if specified theme is not found
    if theme_name not in themes:
        theme_name = "default"
        
    theme = themes[theme_name]
    
    # Apply theme settings
    fig.update_layout(
        plot_bgcolor=theme["bgcolor"],
        paper_bgcolor=theme["bgcolor"],
        font=dict(color=theme["textcolor"]),
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor=theme["gridcolor"],
        zerolinecolor=theme["gridcolor"]
    )
    
    fig.update_yaxes(
        gridcolor=theme["gridcolor"],
        zerolinecolor=theme["gridcolor"]
    )
    
    return fig
