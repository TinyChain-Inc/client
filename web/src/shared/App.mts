import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native-web";

const styles = StyleSheet.create({
  page: {
    minHeight: "100vh",
    padding: 24,
    backgroundColor: "#f3f6fa",
    justifyContent: "center",
    alignItems: "center",
  },
  panel: {
    width: "100%",
    maxWidth: 760,
    backgroundColor: "#ffffff",
    borderColor: "#d5deea",
    borderWidth: 1,
    borderRadius: 14,
    padding: 20,
    gap: 8,
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    color: "#1d2735",
  },
  subtitle: {
    fontSize: 15,
    color: "#52627a",
    marginBottom: 4,
  },
  row: {
    borderColor: "#e8eef7",
    borderWidth: 1,
    borderRadius: 10,
    padding: 10,
    backgroundColor: "#fbfdff",
    gap: 2,
  },
  rowLabel: {
    fontSize: 11,
    color: "#607289",
    textTransform: "uppercase",
    letterSpacing: 0.7,
  },
  rowValue: {
    fontSize: 14,
    color: "#1d2735",
  },
  button: {
    alignSelf: "flex-start",
    marginTop: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: "#0f4bc1",
  },
  buttonPressed: {
    opacity: 0.8,
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "600",
  },
  errorText: {
    marginTop: 4,
    color: "#b61f30",
    fontSize: 13,
  },
});

function getRows(data) {
  return [
    ["source", data?.source ?? "unknown"],
    ["runtime", data?.runtime ?? "unknown"],
    ["package", data?.packageName ?? "none"],
    ["endpoint", data?.endpoint ?? "none"],
    ["scope", data?.scope ?? "none"],
    ["requests", String(data?.requestCount ?? 0)],
    ["message", data?.message ?? "no message"],
    ["fetched", data?.fetchedAt ?? "unknown"],
  ];
}

export function App({ initialData, tinychainClient }) {
  const [data, setData] = React.useState(() => initialData ?? {});
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  const onRefresh = React.useCallback(async () => {
    if (!tinychainClient || typeof tinychainClient.fetchDemoData !== "function") {
      setError("TinyChain client is unavailable in this runtime.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const next = await tinychainClient.fetchDemoData({ scope: "browser-refresh" });
      setData(next);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setLoading(false);
    }
  }, [tinychainClient]);

  const rows = getRows(data);

  return React.createElement(
    View,
    { style: styles.page },
    React.createElement(
      View,
      { style: styles.panel },
      React.createElement(Text, { style: styles.title }, "TinyChain Web SSR/Hydration Demo"),
      React.createElement(
        Text,
        { style: styles.subtitle },
        "Server renders shared React Native Web UI; browser hydrates and calls TinyChain."
      ),
      ...rows.map(([label, value]) =>
        React.createElement(
          View,
          { style: styles.row, key: label },
          React.createElement(Text, { style: styles.rowLabel, testID: `row-label-${label}` }, label),
          React.createElement(Text, { style: styles.rowValue, testID: `row-value-${label}` }, value)
        )
      ),
      React.createElement(
        Pressable,
        {
          style: ({ pressed }) => [styles.button, pressed && styles.buttonPressed],
          onPress: onRefresh,
          accessibilityRole: "button",
          testID: "refresh-button",
        },
        React.createElement(Text, { style: styles.buttonText }, loading ? "Refreshing..." : "Refresh in Browser")
      ),
      error ? React.createElement(Text, { style: styles.errorText }, error) : null
    )
  );
}
