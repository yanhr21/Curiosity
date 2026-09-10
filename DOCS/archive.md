# 历史归档

本次归档：[`legacy/repository_cleanup_20260910/`](../legacy/repository_cleanup_20260910/ARCHIVE.md)。

旧 `PLAN/`、`TODO/`、`DOCS/`（含 project page 与复现长文）、根 README／AGENTS 的
完整工作区版本，以及旧脚本／实验索引，都按原相对路径保留在归档中。未用 Git HEAD
覆盖用户未提交的修改。当前根目录和 DOCS 是重新整理的简短入口。

- [逐项移动映射与元数据](../legacy/repository_cleanup_20260910/MANIFEST.json)
- [旧 PLAN 索引](../legacy/repository_cleanup_20260910/PLAN/README.md)
- [旧 TODO 索引](../legacy/repository_cleanup_20260910/TODO/README.md)
- [旧方法与复现记录](../legacy/repository_cleanup_20260910/DOCS/reproducibility.md)
- [完整旧执行约束与历史](../legacy/repository_cleanup_20260910/AGENTS.md)

133 个过时 demo-following 实验／运行目录、旧 `.runtime`、旧 allocation 记录、
6 个散落运行文件和一个临时生成模块目录一并归档。仍被代码引用的实验依赖、官方
数据／权重、当前 paper Zero-WAM 的所有 endpoint 和保留 GPU 的日志保持原位。

移动使用同文件系统 rename；不删除、不复制大权重、不改写历史 JSON／checkpoint，
不生成文件内容哈希。归档中的旧绝对路径仍是历史记录，按 MANIFEST 的最长匹配 source
前缀替换为 destination 即可定位；旧相对链接先以原位置解析，再应用该映射。

恢复时按映射移动回原位置，必须先确认目标不存在；对已重建的 README／AGENTS／DOCS
先另存当前版本，不能直接覆盖。归档只用于查阅／恢复，不是可直接运行的现行任务队列。
`legacy/` 被 Git 忽略，因此历史备份只在本机存在，不能依赖 fresh clone 找回。
