import escapeHtml from "escape-html";
import serialize from "serialize-javascript";

export { escapeHtml };

export function serializeForInlineScript(value) {
  return serialize(value, { isJSON: true });
}
