"""Run pinned Applio training without unnecessary single-device networking.

The upstream file is not modified. Windows spawn imports this same bootstrap,
so parent and child processes receive the same narrowly scoped AST adapter.
"""
import ast,sys,os
from pathlib import Path

class SingleDeviceTransport(ast.NodeTransformer):
    def __init__(self):self.changed=0
    def visit_Expr(self,node):
        call=node.value
        if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
            and isinstance(call.func.value,ast.Name) and call.func.value.id=='dist'
            and call.func.attr=='init_process_group'):
            self.changed+=1
            condition=ast.parse('n_gpus > 1 and device.type == "cuda"',mode='eval').body
            return ast.copy_location(ast.If(test=condition,body=[node],orelse=[]),node)
        return self.generic_visit(node)

class TraceTraining(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='run':
            body=[]
            for statement in node.body:
                trace=ast.parse(f'print("training stage line {statement.lineno}",flush=True)').body[0]
                body.extend([ast.copy_location(trace,statement),statement])
            node.body=body
        return node

if __name__ in ('__main__','__mp_main__'):
    root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'rvc/train'))
    source=root/'rvc/train/train.py';tree=ast.parse(source.read_text(encoding='utf8'),filename=str(source))
    adapter=SingleDeviceTransport();tree=adapter.visit(tree)
    if adapter.changed!=1:raise RuntimeError('Unexpected upstream training API; run against the pinned backend')
    if os.environ.get('VOICE_COVER_TRAIN_DEBUG')=='1':tree=TraceTraining().visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree,str(source),'exec'),globals())
