import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useAuth } from "@/hooks/use-auth";
import {
  Activity,
  BarChart3,
  ClipboardList,
  Eye,
  Home,
  Info,
  LineChart,
  LogOut,
  Menu,
  Scale,
  ShieldCheck,
} from "lucide-react";
import { NavLink, Outlet, Link } from "react-router";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: Home },
  { to: "/assess", label: "Risk Assessment", icon: ClipboardList },
  { to: "/results", label: "Results", icon: BarChart3 },
  { to: "/analytics", label: "Model Analytics", icon: LineChart },
  { to: "/explain", label: "Explainability", icon: Eye },
  { to: "/responsible-ai", label: "Responsible AI", icon: ShieldCheck },
  { to: "/about", label: "About", icon: Info },
] as const;

export function BrandMark({ className }: { className?: string }) {
  return (
    <Link
      to="/"
      className={cn("flex items-center gap-2.5", className)}
      aria-label="CreditLens home"
    >
      <span className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
        <Scale className="size-5" />
      </span>
      <span className="flex flex-col leading-none">
        <span className="text-[15px] font-semibold tracking-tight">CreditLens</span>
        <span className="mt-0.5 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Loan Risk Assessment
        </span>
      </span>
    </Link>
  );
}

function UserMenu() {
  const { user, isAuthenticated, signOut } = useAuth();
  if (!isAuthenticated) {
    return (
      <Button asChild variant="outline" size="sm">
        <Link to="/auth">Sign in</Link>
      </Button>
    );
  }
  const initials = (user?.name ?? "U")
    .split(/\s+/)
    .map((p) => p[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="rounded-full" aria-label="Account menu">
          <Avatar className="size-8">
            <AvatarFallback className="bg-primary/10 text-primary text-xs font-semibold">
              {initials}
            </AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          <p className="text-sm font-medium">{user?.name ?? "Signed in"}</p>
          <p className="text-xs font-normal text-muted-foreground">
            {user?.email ?? "Authenticated workspace"}
          </p>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link to="/assess">
            <ClipboardList className="size-4" />
            New assessment
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link to="/analytics">
            <LineChart className="size-4" />
            Model analytics
          </Link>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            void signOut();
          }}
          className="text-destructive focus:text-destructive"
        >
          <LogOut className="size-4" />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function AppLayout() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
          <BrandMark />
          <nav className="hidden items-center gap-1 lg:flex" aria-label="Primary">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                    isActive && "bg-accent text-primary"
                  )
                }
              >
                <item.icon className="size-4" />
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <UserMenu />
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" className="lg:hidden" aria-label="Open menu">
                  <Menu className="size-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-80 gap-0 p-0">
                <SheetHeader className="border-b px-5 py-4">
                  <SheetTitle asChild>
                    <BrandMark />
                  </SheetTitle>
                </SheetHeader>
                <nav className="flex flex-col gap-1 p-3" aria-label="Mobile">
                  {NAV_ITEMS.map((item) => (
                    <SheetClose asChild key={item.to}>
                      <NavLink
                        to={item.to}
                        end={item.to === "/"}
                        className={({ isActive }) =>
                          cn(
                            "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                            isActive && "bg-accent text-primary"
                          )
                        }
                      >
                        <item.icon className="size-4" />
                        {item.label}
                      </NavLink>
                    </SheetClose>
                  ))}
                </nav>
                <div className="mt-auto border-t p-4">
                  <Badge variant="secondary" className="gap-1.5">
                    <Activity className="size-3" />
                    Educational ML system
                  </Badge>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-border/70 bg-background">
        <div className="mx-auto grid w-full max-w-7xl gap-10 px-4 py-12 sm:px-6 md:grid-cols-[1.5fr_1fr_1fr]">
          <div>
            <BrandMark />
            <p className="mt-4 max-w-sm text-sm leading-6 text-muted-foreground">
              AI-powered, machine-learning based loan risk assessment and
              approval prediction. Built for education, portfolios, and
              placement demonstrations — not for real lending decisions.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold">Product</h3>
            <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
              <li>
                <Link to="/assess" className="hover:text-foreground">
                  Risk Assessment
                </Link>
              </li>
              <li>
                <Link to="/results" className="hover:text-foreground">
                  Results
                </Link>
              </li>
              <li>
                <Link to="/analytics" className="hover:text-foreground">
                  Model Analytics
                </Link>
              </li>
              <li>
                <Link to="/explain" className="hover:text-foreground">
                  Explainability
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold">Learn</h3>
            <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
              <li>
                <Link to="/responsible-ai" className="hover:text-foreground">
                  Responsible AI
                </Link>
              </li>
              <li>
                <Link to="/about" className="hover:text-foreground">
                  About the project
                </Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-border/70">
          <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 py-5 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <p>
              © {new Date().getFullYear()} CreditLens · Educational project —
              predictions are not financial advice or lending decisions.
            </p>
            <p>Machine Learning · FastAPI-compatible API · Convex backend</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
