/**
 * Model storage: the latest retrained production model lives in the `models`
 * table; predictions and model-info use it when present and otherwise fall
 * back to the bundled artifact. Only model artifacts are stored — never
 * applicant data.
 */

import { internalMutation, internalQuery } from "../_generated/server";
import { v } from "convex/values";

export const getLatestModel = internalQuery({
  args: {},
  handler: async (ctx) => {
    const doc = await ctx.db
      .query("models")
      .withIndex("by_name", (q) => q.eq("name", "production"))
      .order("desc")
      .first();
    if (!doc) return null;
    return doc.payload as unknown;
  },
});

export const storeModel = internalMutation({
  args: { artifact: v.any() },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("models")
      .withIndex("by_name", (q) => q.eq("name", "production"))
      .collect();
    for (const doc of existing) {
      await ctx.db.delete(doc._id);
    }
    await ctx.db.insert("models", {
      name: "production",
      algorithm: String((args.artifact as { selectedModel?: unknown }).selectedModel ?? "random_forest"),
      createdAt: Date.now(),
      payload: args.artifact,
    });
  },
});
