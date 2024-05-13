# Import Libraries
import plotly.graph_objects as go


class Visualizer:
    def __init__(self, model_name: str, data: None, folder_path: str):
        self.model_name = model_name
        self.data = data
        self.folder_path = folder_path

    def plot_ROC(self, title, filename):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data["fpr"],
                y=self.data["tpr"],
                mode="lines",
                name=f'ROC curve (area = {self.data["roc_auc"]:.2f})',
                # line=dict(color="darkorange", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                # line=dict(color="navy", width=2, dash="dash"),
            )
        )
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title=title,
            # legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)"),
            # margin=dict(l=20, r=20, t=30, b=20),
            width=800,
            height=600,
        )
        file_path = self.folder_path + "/ROC_Plot"
        fig.write_image(file_path)

        return file_path

    def plot_F1():
        pass

    def plot_Accuracy():
        pass
