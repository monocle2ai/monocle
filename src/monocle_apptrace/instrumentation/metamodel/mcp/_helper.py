from monocle_apptrace.instrumentation.common.utils import with_tracer_wrapper
from opentelemetry.context import attach, set_value, get_value, detach
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
import json

logger = logging.getLogger(__name__)


def log(arguments):
    print(f"Arguments: {arguments}")


def get_output_text(arguments):
    # arguments["result"].content[0].text
    if (
        "result" in arguments
        and hasattr(arguments["result"], "tools")
        and isinstance(arguments["result"].tools, list)
    ):
        tools = []
        for tool in arguments["result"].tools:
            if hasattr(tool, "name"):
                tools.append(tool.name)
        return tools
    if (
        "result" in arguments
        and hasattr(arguments["result"], "content")
        and isinstance(arguments["result"].content, list)
    ):
        ret_val = []
        for content in arguments["result"].content:
            if hasattr(content, "text"):
                ret_val.append(content.text)
        return ret_val


def get_name(arguments):
    """Get the name of the tool from the instance."""

    args = arguments["args"]
    if (
        args
        and hasattr(args[0], "root")
        and hasattr(args[0].root, "params")
        and hasattr(args[0].root.params, "name")
    ):
        # If the first argument has a root with params and name, return that name
        return args[0].root.params.name


def get_type(arguments):
    """Get the type of the tool from the instance."""
    args = arguments["args"]
    if args and hasattr(args[0], "root") and hasattr(args[0].root, "method"):
        # If the first argument has a root with a method, return that method's name
        return args[0].root.method


def get_params_arguments(arguments):
    """Get the params of the tool from the instance."""

    args = arguments["args"]
    if (
        args
        and hasattr(args[0], "root")
        and hasattr(args[0].root, "params")
        and hasattr(args[0].root.params, "arguments")
    ):
        # If the first argument has a root with params and arguments, return those arguments
        try:
            return json.dumps(args[0].root.params.arguments)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing arguments: {e}")
            return str(args[0].root.params.arguments)


def get_url(arguments):
    """Get the URL of the tool from the instance."""
    url = get_value("mcp.url", None)

    return url

# this extracts the url from the langchain mcp adapter tools and attaches it to the context.
@with_tracer_wrapper
def langchain_mcp_wrapper(
    tracer: any, handler: any, to_wrap, wrapped, instance, source_path, args, kwargs
):
    return_value = None
    try:
        return_value = wrapped(*args, **kwargs)
        return return_value
    finally:
        if (
            return_value
            and hasattr(return_value, "coroutine")
            and kwargs.get("connection", None)
        ):
            try:
                # extract the URL from the connection and attach it to the context
                url = kwargs.get("connection").get("url", None)
                if url:
                    # wrap coroutine methods and attach the URL to the context

                    original_coroutine = return_value.coroutine

                    async def wrapped_coroutine(*args1, **kwargs1):
                        token = None
                        try:
                            token = attach(set_value("mcp.url", url))
                            return await original_coroutine(*args1, **kwargs1)
                        finally:
                            detach(token)

                    return_value.coroutine = wrapped_coroutine

            except Exception as e:
                pass
