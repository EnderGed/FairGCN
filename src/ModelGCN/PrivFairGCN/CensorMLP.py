from ModelSimple.SimpleMLP import SimpleMLPNet


class CensorMLPNet(SimpleMLPNet):
    """
    Three layered MLP model, following "Overlearning reveals sensitive attributes"
    """
    def __init__(self, in_feats, out_feats):
        super().__init__(in_feats=in_feats, out_feats=out_feats, layers=2, dropout=0.1)
