from Bio.PDB import PDBParser, DSSP

# 加载 PDB 文件
parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")

# 调用 DSSP
model = structure[0]  # 获取第一个模型
dssp = DSSP(model, "protein.pdb")

# 提取 ASA 和二级结构
for key in dssp.keys():
    residue = dssp[key]
    sse = residue[2]  # 二级结构
    asa = residue[3]  # 溶剂可接触表面积
    print(f"Residue: {key}, SSE: {sse}, ASA: {asa}")
