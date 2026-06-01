"""Per-leaf response plots produced by export_image with kind="response".

Three private renderers, dispatched by tree task type:

- _plot_regression: per-leaf box [ci_low, ci_high] with a tick at the mean.
- _plot_classification: per (leaf, class) box [ci_low, ci_high] with a
  tick at the class proportion, in the class palette color, with all
  classes of a leaf sharing the same x position and a per-class XY line
  connecting the proportion dots across leaves.
- _plot_survival: one Kaplan-Meier step curve per leaf, on a time x-axis,
  with an optional log-log Greenwood CI band drawn behind each curve when
  the tree's ci_coverage is set.
"""

from __future__ import annotations

import io
import re
import typing

import numpy
import numpy.typing
import scipy.stats

from . import _extension
from . import _node
from . import _palette
from . import _survival
from . import _tree
from . import _tree_classification
from . import _tree_ranking
from . import _tree_regression
from . import _tree_survival
from . import _tree_text

if typing.TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


_LeafT = typing.TypeVar("_LeafT", bound=_node.Node)


# TODO XXX awkward, improve
def _leaf_id(leaf: _LeafT) -> int:
    """Read leaf_id from a Node whose extension is known to be a Leaf."""
    leaf_extension = typing.cast(_extension.Leaf, leaf.extension)
    value = leaf_extension.leaf_id
    return value


def _render_response_image(
    tree: _tree.Tree,
    format: typing.Literal["gif", "pdf", "png", "svg"],
    response_name: None | str,
    class_names: None | list[str],
    dpi: int,
    background_color: None | str,
    displayed_indices: None | list[int],
) -> bytes:
    """Render the per-leaf response plot of tree as image bytes."""
    try:
        import matplotlib.figure
        import matplotlib.backends.backend_agg
    except ImportError as import_error:
        raise ImportError(
            "matplotlib is required for kind='response' image output. "
            "Install it with: pip install ars-sigma[viz]"
        ) from import_error
    figure = matplotlib.figure.Figure(figsize=(8.0, 4.5), dpi=dpi)
    matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    if isinstance(tree, _tree_regression.RegressionTree):
        _plot_regression(axes, tree, response_name)
    elif isinstance(tree, _tree_classification.ClassificationTree):
        _plot_classification(axes, tree, class_names)
    elif isinstance(tree, _tree_survival.SurvivalTree):
        _plot_survival(axes, tree, response_name)
    elif isinstance(tree, _tree_ranking.RankingTree):
        ranking_indices = [] if displayed_indices is None else displayed_indices
        _plot_ranking(axes, tree, ranking_indices)
    else:
        raise TypeError(f"unsupported tree type: {type(tree).__name__}")
    transparent = background_color == "transparent"
    if transparent:
        figure.patch.set_alpha(0.0)
        axes.patch.set_alpha(0.0)
    elif background_color is not None:
        figure.patch.set_facecolor(background_color)
    figure.tight_layout()
    payload = _save_figure(figure, format, dpi, transparent)
    return payload


def _save_figure(
    figure: matplotlib.figure.Figure,
    format: typing.Literal["gif", "pdf", "png", "svg"],
    dpi: int,
    transparent: bool,
) -> bytes:
    """Serialize figure to the requested image format and return its bytes."""
    match format:
        case "gif":
            png_buffer = io.BytesIO()
            figure.savefig(
                png_buffer, format="png", dpi=dpi, transparent=transparent
            )
            png_buffer.seek(0)
            import PIL.Image

            image = PIL.Image.open(png_buffer)
            gif_buffer = io.BytesIO()
            image.save(gif_buffer, format="GIF")
            payload = gif_buffer.getvalue()
            return payload
        case "svg":
            buffer = io.BytesIO()
            figure.savefig(
                buffer, format=format, dpi=dpi, transparent=transparent
            )
            payload = _scale_svg_root_dimensions(buffer.getvalue(), dpi)
            return payload
        case _:
            buffer = io.BytesIO()
            figure.savefig(
                buffer, format=format, dpi=dpi, transparent=transparent
            )
            payload = buffer.getvalue()
            return payload


def _scale_svg_root_dimensions(svg_bytes: bytes, dpi: int) -> bytes:
    """Multiply the root <svg> width/height pt attributes by dpi/72."""
    decoded = svg_bytes.decode("utf-8")
    open_match = re.search(r"<svg\b[^>]*>", decoded)
    if open_match is None:
        return svg_bytes
    factor = dpi / 72.0
    original_tag = open_match.group(0)

    def scale(attribute_match: re.Match[str]) -> str:
        attribute = attribute_match.group(1)
        value = float(attribute_match.group(2))
        scaled = value * factor
        replacement = f'{attribute}="{scaled:g}pt"'
        return replacement

    new_tag = re.sub(r'(width|height)="([0-9.]+)pt"', scale, original_tag)
    head = decoded[: open_match.start()]
    tail = decoded[open_match.end() :]
    result = head + new_tag + tail
    encoded = result.encode("utf-8")
    return encoded


def _plot_regression(
    axes: matplotlib.axes.Axes,
    tree: _tree_regression.RegressionTree,
    response_name: None | str,
) -> None:
    """Draw per-leaf raincloud + mean-with-CI for a regression tree."""
    leaves = tree.leaves_
    n_leaves = len(leaves)
    width = 0.6
    for leaf in leaves:
        leaf_id = _leaf_id(leaf)
        x = leaf_id + 1
        color = _palette._leaf_color(leaf_id, n_leaves)
        _draw_response_raincloud(
            axes, x, leaf.response_samples, color, tree.random_state
        )
        ci_low = leaf.ci_low
        ci_high = leaf.ci_high
        if ci_low is not None and ci_high is not None:
            height = ci_high - ci_low
            axes.bar(
                x,
                height=height,
                bottom=ci_low,
                width=width,
                color=color,
                alpha=0.4,
                edgecolor=color,
                linewidth=1.0,
                zorder=2,
            )
        prediction = leaf.prediction
        axes.hlines(
            prediction,
            x - width / 2.0,
            x + width / 2.0,
            colors=color,
            linewidth=2.0,
            zorder=3,
        )
        axes.scatter([x], [prediction], color=color, s=12, zorder=4)
    _configure_leaf_x_axis(axes, n_leaves)
    if response_name is None:
        y_label = "Response mean"
    else:
        y_label = f"{_tree_text._capitalize_first_letter(response_name)} mean"
    axes.set_ylabel(y_label)
    axes.grid(axis="y", linestyle=":", alpha=0.4)


def _draw_response_raincloud(
    axes: matplotlib.axes.Axes,
    x: int,
    samples: numpy.typing.NDArray[numpy.floating],
    color: str,
    random_state: None | int,
) -> None:
    """Overlay a per-leaf raincloud (half-violin + box + jittered dots).

    Drawn at integer x-coordinate, with every shape in the given color.
    The violin renders to the right of x, the boxplot is centered on x,
    and the raw dots scatter to the left of x. Jitter is seeded by the
    fitted tree's random_state combined with the leaf column index, so
    plots with the same random_state are byte-identical while adjacent
    leaves still receive distinct jitter patterns. Skipped silently when
    fewer than two samples are available (KDE requires at least two
    distinct points).
    """
    import matplotlib.patches

    if samples.size == 0:
        return
    samples = numpy.asarray(samples, dtype=float)
    if samples.size >= 2 and float(numpy.ptp(samples)) > 0.0:
        kde = scipy.stats.gaussian_kde(samples)
        margin = 0.05 * float(numpy.ptp(samples))
        grid = numpy.linspace(
            float(samples.min()) - margin,
            float(samples.max()) + margin,
            200,
        )
        density = kde(grid)
        density_max = float(density.max())
        scale = 0.35 / density_max if density_max > 0.0 else 0.0
        right_edge = x + density * scale
        axes.fill_betweenx(
            grid,
            x,
            right_edge,
            color=color,
            alpha=0.25,
            linewidth=0.0,
            zorder=1,
        )
    box_half_width = 0.05
    quartiles = numpy.quantile(samples, [0.25, 0.5, 0.75])
    q1, median, q3 = (
        float(quartiles[0]),
        float(quartiles[1]),
        float(quartiles[2]),
    )
    iqr = q3 - q1
    whisker_low = float(
        numpy.min(samples[samples >= q1 - 1.5 * iqr], initial=q1)
    )
    whisker_high = float(
        numpy.max(samples[samples <= q3 + 1.5 * iqr], initial=q3)
    )
    axes.add_patch(
        matplotlib.patches.Rectangle(
            (x - box_half_width, q1),
            2.0 * box_half_width,
            iqr,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=2,
        )
    )
    axes.hlines(
        median,
        x - box_half_width,
        x + box_half_width,
        colors=color,
        linewidth=1.5,
        zorder=3,
    )
    axes.vlines(
        x,
        whisker_low,
        q1,
        colors=color,
        linewidth=1.0,
        zorder=2,
    )
    axes.vlines(
        x,
        q3,
        whisker_high,
        colors=color,
        linewidth=1.0,
        zorder=2,
    )
    if random_state is None:
        rng = numpy.random.default_rng(int(x))
    else:
        seed_seq = numpy.random.SeedSequence(random_state, spawn_key=(int(x),))
        rng = numpy.random.default_rng(seed_seq)
    jitter = rng.uniform(-0.18, -0.02, size=samples.size)
    axes.scatter(
        x + jitter,
        samples,
        color=color,
        s=4,
        alpha=0.5,
        linewidths=0.0,
        zorder=2,
    )


def _plot_classification(
    axes: matplotlib.axes.Axes,
    tree: _tree_classification.ClassificationTree,
    class_names: None | list[str],
) -> None:
    """Draw per (leaf, class) class-proportion dots with CI boxes.

    Per (leaf, class): a translucent CI box, a horizontal tick at the
    class proportion, and a marker dot, all in the class palette color.
    All classes of a leaf share the same x position; a per-class line
    connects the proportion dots across leaves.
    """
    import matplotlib.patches

    leaves = tree.leaves_
    n_leaves = len(leaves)
    n_classes = int(tree.n_classes_)
    labels = _resolve_class_labels(tree, class_names)
    bar_width = 0.6
    sorted_leaves = sorted(leaves, key=_leaf_id)
    line_xs = [_leaf_id(leaf) + 1 for leaf in sorted_leaves]
    if tree.reverse_order:
        class_order = list(range(n_classes - 1, -1, -1))
    else:
        class_order = list(range(n_classes))
    for slot_idx, class_index in enumerate(class_order):
        color = _palette._leaf_color(slot_idx, n_classes)
        for leaf in leaves:
            leaf_id = _leaf_id(leaf)
            x = leaf_id + 1
            proportion = float(leaf.class_distribution[class_index]) * 100.0
            ci_low = leaf.ci_low
            ci_high = leaf.ci_high
            if ci_low is not None and ci_high is not None:
                ci_low_k = float(ci_low[class_index]) * 100.0
                ci_high_k = float(ci_high[class_index]) * 100.0
                height = ci_high_k - ci_low_k
                axes.bar(
                    x,
                    height=height,
                    bottom=ci_low_k,
                    width=bar_width,
                    color=color,
                    alpha=0.4,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=2,
                )
            axes.hlines(
                proportion,
                x - bar_width / 2.0,
                x + bar_width / 2.0,
                colors=color,
                linewidth=2.0,
                zorder=3,
            )
            axes.scatter([x], [proportion], color=color, s=12, zorder=4)
        line_ys = [
            float(leaf.class_distribution[class_index]) * 100.0
            for leaf in sorted_leaves
        ]
        axes.plot(line_xs, line_ys, color=color, linewidth=1.5, zorder=3)
    _configure_leaf_x_axis(axes, n_leaves)
    axes.set_ylim(0.0, 100.0)
    axes.set_ylabel("Class probability, %")
    axes.grid(axis="y", linestyle=":", alpha=0.4)
    handles = [
        matplotlib.patches.Patch(
            color=_palette._leaf_color(slot_idx, n_classes),
            label=labels[class_index],
        )
        for slot_idx, class_index in enumerate(class_order)
    ]
    axes.legend(
        handles=handles,
        title="Class",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _plot_survival(
    axes: matplotlib.axes.Axes,
    tree: _tree_survival.SurvivalTree,
    response_name: None | str,
) -> None:
    """Draw one Kaplan-Meier step curve per leaf for a survival tree.

    When the tree's ci_coverage is not None, also draws a pointwise log-log
    Greenwood confidence band behind each curve, in the same color at low
    opacity.
    """
    leaves = tree.leaves_
    n_leaves = len(leaves)
    for leaf in leaves:
        leaf_id = _leaf_id(leaf)
        badge_number = leaf_id + 1
        color = _palette._leaf_color(leaf_id, n_leaves)
        times, surv = leaf.survival_function
        times_with_origin = numpy.concatenate(
            [[0.0], numpy.asarray(times, dtype=float)]
        )
        surv_with_origin = numpy.concatenate(
            [[1.0], numpy.asarray(surv, dtype=float)]
        )
        ci_coverage = tree.ci_coverage
        if ci_coverage is not None and len(times) > 0:
            alpha = (1.0 - ci_coverage) / 2.0
            ci_low, ci_high = _survival.compute_log_log_ci_band(
                numpy.asarray(surv, dtype=float),
                leaf.survival_log_variance,
                alpha,
            )
            ci_low_with_origin = numpy.concatenate([[1.0], ci_low])
            ci_high_with_origin = numpy.concatenate([[1.0], ci_high])
            axes.fill_between(
                times_with_origin,
                ci_low_with_origin * 100.0,
                ci_high_with_origin * 100.0,
                step="post",
                color=color,
                alpha=0.15,
                linewidth=0.0,
                zorder=1,
            )
        axes.step(
            times_with_origin,
            surv_with_origin * 100.0,
            where="post",
            color=color,
            label=str(badge_number),
        )
        axes.scatter(
            [times_with_origin[0], times_with_origin[-1]],
            [surv_with_origin[0] * 100.0, surv_with_origin[-1] * 100.0],
            color=color,
            s=12,
            zorder=3,
        )
    axes.set_ylim(-2.0, 102.0)
    axes.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    axes.set_ylabel("Survival probability, %")
    if response_name is None:
        x_label = "Time"
    else:
        x_label = _tree_text._capitalize_first_letter(response_name)
    axes.set_xlabel(x_label)
    axes.grid(linestyle=":", alpha=0.4)
    handles, legend_labels = axes.get_legend_handles_labels()
    order = list(range(len(handles)))[::-1]
    axes.legend(
        [handles[index] for index in order],
        [legend_labels[index] for index in order],
        title="Leaf number",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _plot_ranking(
    axes: matplotlib.axes.Axes,
    tree: _tree_ranking.RankingTree,
    displayed_indices: list[int],
) -> None:
    """Draw per (leaf, item) mean-rank dots with CI boxes.

    Per (leaf, item): a translucent CI box, a horizontal tick at the mean
    rank, and a marker dot, all in the per-item palette color. All items
    of a leaf share the same x position; a per-item line connects the
    mean-rank dots across leaves.
    """
    import matplotlib.patches

    leaves = tree.leaves_
    n_leaves = len(leaves)
    item_names = [str(name) for name in tree.item_names_]
    bar_width = 0.6
    sorted_leaves = sorted(leaves, key=_leaf_id)
    line_xs = [_leaf_id(leaf) + 1 for leaf in sorted_leaves]
    if tree.reverse_order:
        item_order = list(reversed(displayed_indices))
    else:
        item_order = list(displayed_indices)
    displayed_count = len(item_order)
    for slot_idx, item_index in enumerate(item_order):
        color = _palette._leaf_color(slot_idx, displayed_count)
        for leaf in leaves:
            leaf_id = _leaf_id(leaf)
            x = leaf_id + 1
            metric = leaf.metrics[item_index]
            mean_rank = float(metric.value)
            if numpy.isnan(mean_rank):
                continue
            if metric.ci_low is not None and metric.ci_high is not None:
                low = float(metric.ci_low)
                high = float(metric.ci_high)
                if not (numpy.isnan(low) or numpy.isnan(high)):
                    height = high - low
                    axes.bar(
                        x,
                        height=height,
                        bottom=low,
                        width=bar_width,
                        color=color,
                        alpha=0.4,
                        edgecolor=color,
                        linewidth=1.0,
                        zorder=2,
                    )
            axes.hlines(
                mean_rank,
                x - bar_width / 2.0,
                x + bar_width / 2.0,
                colors=color,
                linewidth=2.0,
                zorder=3,
            )
            axes.scatter([x], [mean_rank], color=color, s=12, zorder=4)
        line_ys = [
            float(leaf.metrics[item_index].value) for leaf in sorted_leaves
        ]
        axes.plot(line_xs, line_ys, color=color, linewidth=1.5, zorder=3)
    _configure_leaf_x_axis(axes, n_leaves)
    all_values: list[float] = []
    for leaf in leaves:
        for item_index in item_order:
            value = leaf.metrics[item_index].value
            if not numpy.isnan(value):
                all_values.append(value)
            metric = leaf.metrics[item_index]
            if metric.ci_low is not None and not numpy.isnan(metric.ci_low):
                all_values.append(float(metric.ci_low))
            if metric.ci_high is not None and not numpy.isnan(metric.ci_high):
                all_values.append(float(metric.ci_high))
    if all_values:
        lower = min(all_values)
        upper = max(all_values)
        padding = max(1.0, 0.05 * (upper - lower))
        axes.set_ylim(lower - padding, upper + padding)
    axes.invert_yaxis()
    axes.set_ylabel("Mean rank (1 = most preferred)")
    axes.grid(axis="y", linestyle=":", alpha=0.4)
    handles = [
        matplotlib.patches.Patch(
            color=_palette._leaf_color(slot_idx, displayed_count),
            label=_tree_text._capitalize_first_letter(item_names[item_index]),
        )
        for slot_idx, item_index in enumerate(item_order)
    ]
    axes.legend(
        handles=handles,
        title="Item",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _configure_leaf_x_axis(axes: matplotlib.axes.Axes, n_leaves: int) -> None:
    """Set integer leaf-number ticks 1..n_leaves and the x-axis label."""
    ticks = list(range(1, n_leaves + 1))
    axes.set_xticks(ticks)
    axes.set_xticklabels([str(t) for t in ticks])
    axes.set_xlim(0.5, n_leaves + 0.5)
    axes.set_xlabel("Leaf number")


def _resolve_class_labels(
    tree: _tree_classification.ClassificationTree, class_names: None | list[str]
) -> list[str]:
    """Return display labels per class, falling back to tree.classes_."""
    if class_names is None:
        raw_labels = [str(c) for c in tree.classes_]
    else:
        raw_labels = list(class_names)
    labels = [
        _tree_text._capitalize_first_letter(label) for label in raw_labels
    ]
    return labels
