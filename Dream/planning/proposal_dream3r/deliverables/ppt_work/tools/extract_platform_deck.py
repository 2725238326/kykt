from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw
import win32com.client


BASE = Path(r"E:\kykt\Dream\planning\proposal_dream3r\deliverables\ppt_work")
SRC = BASE / "proposal_dream3r_opening_report_final_text_only_cleaned_with_notes.pptx"
OUT = BASE / "proposal_dream3r_platform_section_only.pptx"
PREVIEW_DIR = BASE / "previews_platform_section_only"
CONTACT = BASE / "contact_sheet_platform_section_only.png"


def find_shape(slide, shape_id: int):
    for idx in range(1, slide.Shapes.Count + 1):
        shape = slide.Shapes(idx)
        if shape.Id == shape_id:
            return shape
    raise RuntimeError(f"shape id={shape_id} not found on slide {slide.SlideIndex}")


def set_text(slide, shape_id: int, text: str, size: float | None = None) -> None:
    shape = find_shape(slide, shape_id)
    tr = shape.TextFrame.TextRange
    tr.Text = text
    if size is not None:
        tr.Font.Size = size


def set_notes(slide, text: str) -> None:
    body = None
    for idx in range(1, slide.NotesPage.Shapes.Count + 1):
        shape = slide.NotesPage.Shapes(idx)
        try:
            if shape.PlaceholderFormat.Type == 2:
                body = shape
                break
        except Exception:
            pass
    if body is None:
        body = slide.NotesPage.Shapes.AddTextbox(1, 54, 346.5, 432, 283.5)
    body.TextFrame.TextRange.Text = text
    body.TextFrame.TextRange.Font.Name = "微软雅黑"
    body.TextFrame.TextRange.Font.Size = 11


def export_contact() -> None:
    def slide_no(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else 0

    files = sorted(PREVIEW_DIR.glob("*.PNG"), key=slide_no)
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((360, 203), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 223), "white")
        canvas.paste(im, (0, 20))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, 359, 222], outline=(210, 220, 230))
        draw.text((8, 3), f"Slide {slide_no(f):02d}", fill=(0, 70, 130))
        thumbs.append(canvas)

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 360, rows * 223), (245, 247, 250))
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 360, (idx // cols) * 223))
    sheet.save(CONTACT)


def main() -> None:
    shutil.copy2(SRC, OUT)
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Open(str(OUT), WithWindow=False)

    keep = {1, 2, 13, 14, 15, 16, 17, 18, 24}
    for i in range(pres.Slides.Count, 0, -1):
        if i not in keep:
            pres.Slides(i).Delete()

    # New slide mapping:
    # 1 cover, 2 outline, 3 old13, 4 old14, 5 old15, 6 old16, 7 old17, 8 old18, 9 old24
    set_text(pres.Slides(1), 2, "面向前馈式三维重建的\n聚合管理平台设计与实现", 34)

    s = pres.Slides(2)
    set_text(s, 8, "平台建设背景")
    set_text(s, 12, "核心功能设计")
    set_text(s, 16, "技术架构与实测")
    set_text(s, 20, "后续应用方向")

    s = pres.Slides(3)
    set_text(s, 90, "01")
    set_text(s, 3, "学习、实践和实验验证都需要更低门槛的部署与算力调度。", 20)
    set_text(s, 4, "部署门槛高")
    set_text(s, 5, "模型环境、权重路径和推理脚本各不相同")
    set_text(s, 6, "算力使用分散")
    set_text(s, 7, "上传、启动、日志和结果回传依赖手动命令")
    set_text(s, 8, "对比困难")
    set_text(s, 9, "输入输出格式不统一，横向比较需大量手动对齐")
    set_text(s, 10, "下游调用不便")
    set_text(s, 11, "模型能力停留在脚本级，缺少标准接口供应用集成")
    set_text(s, 12, "快速部署、算力调度、统一对比与 API 封装")

    s = pres.Slides(4)
    set_text(s, 90, "02")
    set_text(s, 3, "平台覆盖从模型接入、远端运行到结果输出的完整链路。", 20)
    set_text(s, 5, "一键部署\r提交任务\r查看状态")
    set_text(s, 7, "远端运行\r日志跟踪\r结果回传")
    set_text(s, 11, "执行合同\r新模型快速上线")
    set_text(s, 14, "部署→调度→对比→输出，服务学习实践和研究实验")

    s = pres.Slides(5)
    set_text(s, 90, "02")
    set_text(s, 3, "平台通过统一执行合同，把桌面操作、本地管理和远端算力连接起来。", 20)
    set_text(s, 6, "执行器封装模型差异，平台统一提交、日志和结果回传，顶层预留 API 导出接口。")

    s = pres.Slides(6)
    set_text(s, 90, "02")
    set_text(s, 3, "6 个模型已完成端到端验证，统一运行记录可支撑耗时和算力预算比较。", 20)
    set_text(s, 13, "统一条件下记录耗时差异，为模型选择和算力安排提供依据")

    s = pres.Slides(7)
    set_text(s, 90, "03")
    set_text(s, 3, "平台后续从学习实践工具，逐步扩展为模型能力输出通道。", 20)
    set_text(s, 5, "学习与部署：完善任务视图，降低模型试用成本")
    set_text(s, 7, "实验与对比：新模型一键接入、即时对比，结果直接导出")
    set_text(s, 11, "课程实践 → 研究实验 → 下游应用集成")

    s = pres.Slides(8)
    set_text(s, 90, "03")
    set_text(s, 92, "平台如何支撑研究与实践")
    set_text(s, 3, "平台既支撑架构验证，也支撑新模型的部署和算力使用管理。", 20)
    set_text(s, 4, "研究与实践需求")
    set_text(s, 9, "快速部署\r算力调度\r多模型对照\r新模型接入")
    set_text(s, 13, "平台先支撑部署与验证，后续可把稳定模型封装为 API 对外输出")

    s = pres.Slides(9)
    set_text(s, 4, "本平台面向前馈式三维重建模型，降低学习和实践中的部署成本，统一管理远端算力、模型运行和结果对比。\r后续将继续补齐快速部署、报告导出和 API 化调用能力。")

    notes = {
        1: "封面\n\n各位老师好，我汇报的内容是面向前馈式三维重建的聚合管理平台设计与实现。这个平台的目标，是让 3R 模型在学习、实践和实验验证中更容易部署、更容易运行，也更方便使用远端算力。",
        2: "汇报提纲\n\n我主要讲四部分：第一，为什么需要这样一个平台；第二，平台目前具备哪些核心功能；第三，技术架构和已接入模型的实测情况；第四，平台后续如何继续扩展到应用方向。",
        3: "为什么需要一个聚合平台\n\n做这个平台首先是为了解决学习和实践中的部署困难。3R 模型各自有不同环境、依赖、权重和脚本，新手想跑起来成本很高。第二是算力使用分散，远端 GPU 上的上传、启动、日志查看和结果回传都依赖手动命令。第三是模型之间的输入输出格式不统一，横向对比很麻烦。平台要把这些环节统一起来，形成快速部署、算力调度、统一对比和 API 封装的基础。",
        4: "平台核心功能\n\n平台功能围绕完整链路设计。命令中心提供统一入口，任务工作台负责远端运行、日志跟踪和结果回传，样本矩阵支持同一输入下的多模型并排对比。模型注册通过执行合同降低新模型接入成本，能力输出负责统一格式导出，并为后续 API 封装预留接口。",
        5: "技术架构\n\n技术上分为桌面前端、本地后端、远端调度和模型执行器四层。前端负责交互，本地后端管理模型注册和任务队列，远端调度通过 SSH 和 SCP 使用 GPU 机器，执行器只封装每个模型自己的推理差异。这样使用者不需要每次手动记命令和路径，平台可以把算力使用过程标准化。",
        6: "已接入模型与实测数据\n\n目前平台已经完成 6 个模型的端到端验证。右侧耗时图说明，不同模型在统一输入和统一机器下速度差异明显。这个数据的意义不只是比较快慢，也帮助我们后续做算力预算和模型选择：什么时候用快模型，什么时候值得用更慢但能力更强的模型。",
        7: "后续更新方向与应用场景\n\n后续平台分三步推进。短期先把学习和部署体验做好，降低模型试用成本。中期支撑实验和对比，新模型接入后能够快速运行、即时比较，结果直接导出。长期则考虑 API 化服务，把验证过的模型能力提供给 SLAM、AR 和机器人场景理解等下游任务。",
        8: "平台如何支撑研究与实践\n\n平台既支撑课题里的架构验证，也支撑日常学习和实践部署。研究上，它提供统一输入、统一输出和同条件对比；实践上，它提供远端算力调度、任务记录和新模型接入。后续新架构训练完成后，也可以作为一个模型接入平台，继续走同样的验证和输出流程。",
        9: "总结致谢\n\n总结一下，这个平台的核心价值是降低 3R 模型学习和实践中的部署成本，把模型运行、算力使用和结果对比统一起来。后续会继续补齐快速部署、报告导出和 API 化调用能力。以上是我的汇报，请各位老师批评指正。",
    }
    for i, note in notes.items():
        set_notes(pres.Slides(i), note)

    pres.Save()
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pres.Export(str(PREVIEW_DIR), "PNG", 1920, 1080)
    pres.Close()
    app.Quit()

    export_contact()
    print(OUT)
    print(PREVIEW_DIR)
    print(CONTACT)


if __name__ == "__main__":
    main()
