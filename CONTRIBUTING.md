# 贡献指南 / Contributing Guide

感谢你考虑为本项目做出贡献！以下是详细的贡献指南。

Thank you for considering contributing to this project! Here is a detailed contribution guide.

## 📋 贡献前准备 / Before Contributing

1. **熟悉 Git 基础操作** / Familiarize yourself with basic Git operations
2. **阅读本文档** / Read this document
3. **查看已有资料，避免重复上传** / Check existing materials to avoid duplicates

## 🔄 贡献流程 / Contribution Process

### 1. Fork 项目 / Fork the Project

点击右上角的 "Fork" 按钮，将项目 Fork 到你的账号下。

Click the "Fork" button in the upper right corner to fork the project to your account.

### 2. 克隆仓库 / Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Review-documents-for-math-in-DLUT.git
cd Review-documents-for-math-in-DLUT
```

### 3. 创建分支 / Create a Branch

```bash
git checkout -b add-course-materials
```

分支命名建议 / Branch naming suggestions:
- `add-math-analysis-notes` - 添加数学分析笔记
- `update-algebra-solutions` - 更新代数习题解答
- `fix-typo-readme` - 修复 README 错误

### 4. 添加资料 / Add Materials

将你的资料放入对应的文件夹中。如果文件夹不存在，请创建它。

Place your materials in the appropriate folder. If the folder doesn't exist, please create it.

#### 文件夹命名规范 / Folder Naming Convention

```
年级/学期/课程名称/资料类型/
例如: 大一/数学分析/期末复习/
```

#### 文件命名规范 / File Naming Convention

```
课程名称-资料类型-年份/学期.格式
例如: 数学分析-期末试卷-2024春.pdf
      Mathematical_Analysis-Final_Exam-2024Spring.pdf
```

### 5. 提交改动 / Commit Changes

```bash
git add .
git commit -m "添加数学分析期末复习资料"
```

提交信息规范 / Commit Message Convention:
- `添加 [课程名] [资料类型]` / `Add [Course] [Material Type]`
- `更新 [课程名] [资料类型]` / `Update [Course] [Material Type]`
- `修复 [问题描述]` / `Fix [Issue Description]`

### 6. 推送到远程 / Push to Remote

```bash
git push origin add-course-materials
```

### 7. 创建 Pull Request / Create Pull Request

1. 访问你 Fork 的仓库页面
2. 点击 "Pull Request" 按钮
3. 填写 PR 描述，说明你添加的内容
4. 提交 PR 等待审核

## 📝 资料要求 / Material Requirements

### 可接受的资料 / Acceptable Materials

✅ 个人整理的课程笔记
✅ 自己完成的习题解答
✅ 知识点总结和复习提纲
✅ 学习经验和心得分享
✅ 已公开的往年试卷（注明来源）

### 不可接受的资料 / Unacceptable Materials

❌ 未经授权的教师课件
❌ 未公开的试卷答案
❌ 侵犯版权的教材扫描件
❌ 包含个人隐私信息的文件
❌ 未注明来源的他人作品

## 📐 格式规范 / Format Guidelines

### 文件格式 / File Formats

推荐格式 / Recommended formats:
- **PDF**: 适合笔记、试卷等静态文档
- **Markdown**: 适合文本型资料，便于版本控制
- **LaTeX**: 适合数学公式较多的文档（请同时提供 PDF）

不推荐格式 / Not recommended:
- Word 文档（.doc, .docx）- 请转换为 PDF
- 图片格式作为主要文档（除非是扫描件）

### 文件大小 / File Size

- **小于 10MB**: 直接上传到 Git
- **10MB - 100MB**: 使用 Git LFS
- **大于 100MB**: 建议上传到网盘，提供链接

Git LFS 使用方法 / Git LFS Usage:
```bash
git lfs install
git lfs track "*.pdf"
git add .gitattributes
```

### 文档结构 / Document Structure

每个课程文件夹应包含以下内容（如有）：

Each course folder should contain the following (if available):

```
课程名/
├── README.md           # 课程简介和资料目录
├── 笔记/Notes/         # 课程笔记
├── 习题/Exercises/     # 习题及解答
├── 试卷/Exams/         # 往年试卷
└── 总结/Summaries/     # 知识点总结
```

## ✅ 质量标准 / Quality Standards

### 笔记类资料 / Notes

- 内容完整，逻辑清晰
- 公式准确，排版整洁
- 有必要的说明和注释

### 习题解答 / Exercise Solutions

- 解答过程完整
- 步骤清晰易懂
- 注明题目来源

### 知识点总结 / Knowledge Summaries

- 覆盖重要知识点
- 结构化组织
- 便于快速复习

## 🔍 审核标准 / Review Criteria

PR 将根据以下标准进行审核：

PRs will be reviewed based on the following criteria:

1. **合规性** / Compliance
   - 不侵犯版权
   - 符合学术诚信原则

2. **完整性** / Completeness
   - 文件命名规范
   - 目录结构清晰

3. **质量** / Quality
   - 内容准确
   - 格式规范

4. **实用性** / Usefulness
   - 对学习有帮助
   - 内容不重复

## 🐛 报告问题 / Report Issues

如果发现仓库中的问题，请：

If you find issues in the repository, please:

1. 在 Issues 中创建新问题
2. 使用清晰的标题描述问题
3. 提供详细的问题描述
4. 如果可能，提出解决方案

## 💡 建议 / Suggestions

欢迎提出以下建议：

We welcome suggestions on:

- 仓库结构优化
- 新的资料类型
- 改进文档组织方式
- 其他有助于项目发展的想法

## 📞 联系方式 / Contact

如有疑问，可以通过以下方式联系：

If you have questions, you can contact us via:

- 创建 Issue
- 在 Pull Request 中评论
- 在 Discussion 中讨论

## 🙏 致谢 / Acknowledgments

再次感谢你的贡献！你的努力将帮助更多的同学。

Thank you again for your contribution! Your effort will help more students.

---

**记住**: 质量比数量更重要。一份精心整理的资料胜过十份草率的笔记。

**Remember**: Quality over quantity. One well-organized material is better than ten hastily written notes.
